"""Evaluate compression at different compression ratios (CR).

Week 2 Person A deliverable:
- Load trained model from Week 1
- Evaluate at different CRs (achieved via quantization and/or latent_dim)
- Generate PRD-CR and WWPRD-CR data
- Save results to JSON for Person B visualization

Author: Person A
Date: October 2025
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import track
from rich.table import Table

from ecgdae.data import MITBIHDataset, NSTDBNoiseMixer, WindowingConfig, gaussian_snr_mixer
from ecgdae.models import ConvAutoEncoder, ResidualAutoEncoder
from ecgdae.metrics import (
    compute_prd, compute_wwprd, compute_snr,
    compute_derivative_weights, batch_evaluate
)
from ecgdae.quantization import (
    quantize_latent, dequantize_latent,
    compute_compression_ratio
)

console = Console()


def load_model(model_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint."""
    # Create model
    if config['model_type'] == 'conv':
        model = ConvAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(config['hidden_dims']),
            latent_dim=config['latent_dim'],
        )
    else:
        model = ResidualAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(config['hidden_dims']),
            latent_dim=config['latent_dim'],
            num_res_blocks=2,
        )

    # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume checkpoint is directly the state dict
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    console.print(f"[green]✓ Loaded model from {model_path}")

    return model


def create_test_dataloader(config: dict, num_test_samples: int = 500) -> DataLoader:
    """Create test dataloader for evaluation."""
    # Setup windowing
    window_config = WindowingConfig(
        sample_rate=config['sample_rate'],
        window_seconds=config['window_seconds'],
        step_seconds=config['window_seconds'],
    )

    # Setup noise mixer
    if config['noise_type'] == 'gaussian':
        noise_mixer = gaussian_snr_mixer(config['snr_db'])
    else:
        nstdb = NSTDBNoiseMixer(data_dir="./data/nstdb")
        noise_mixer = nstdb.create_mixer(
            target_snr_db=config['snr_db'],
            noise_type=config['nstdb_noise'],
        )

    # Use different records for testing (records after training set)
    from ecgdae.data import MITBIHLoader
    all_records = MITBIHLoader.MITBIH_RECORDS
    num_train_records = config.get('num_records', 10)
    test_records = all_records[num_train_records:num_train_records + 5]

    console.print(f"[cyan]Test records: {test_records}")

    # Create dataset
    dataset = MITBIHDataset(
        records=test_records,
        config=window_config,
        noise_mixer=noise_mixer,
        data_dir=config.get('data_dir', './data/mitbih'),
        channel=0,
        normalize=True,
    )

    # Limit dataset size for faster evaluation
    indices = torch.randperm(len(dataset))[:num_test_samples]
    from torch.utils.data import Subset
    subset = Subset(dataset, indices.tolist())

    loader = DataLoader(
        subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    return loader


@torch.no_grad()
def evaluate_at_cr(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_cr: float,
    quantization_bits: int = 8,
    weight_alpha: float = 2.0,
    sample_rate: int = 360,
    window_length: int = 512,
) -> Dict[str, float]:
    """Evaluate model at a specific compression ratio.

    Achieves target CR by quantizing latent representation to different bit depths.
    The actual CR is computed from the quantized latent size.

    Args:
        model: Trained autoencoder model
        dataloader: Test data loader
        device: Torch device
        target_cr: Target compression ratio (used to select quantization_bits if not specified)
        quantization_bits: Bits for quantization (4, 6, or 8)
        weight_alpha: Alpha for WWPRD weights
        sample_rate: ECG sampling rate
        window_length: Signal window length

    Returns:
        Dictionary of metrics: PRD, WWPRD, SNR_in, SNR_out, SNR_improvement, etc.
    """
    model.eval()

    all_prds = []
    all_wwprds = []
    all_snr_ins = []
    all_snr_outs = []
    all_snr_improvements = []
    all_actual_crs = []

    console.print(f"[yellow]Evaluating at target CR={target_cr:.1f}:1 (quantization_bits={quantization_bits})...")

    for noisy, clean in track(dataloader, description=f"  Processing"):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Encode to latent
        latent = model.encode(noisy)

        # Quantize latent representation
        quantized, metadata = quantize_latent(latent, quantization_bits, return_metadata=True)

        # Calculate actual compression ratio for this batch
        # Original size: window_length samples * 11 bits/sample
        original_bits = window_length * 11
        # Compressed size: latent shape (B, C, T) quantized with quantization_bits
        latent_shape = latent.shape  # (B, C, T)
        latent_size = latent_shape[1] * latent_shape[2]  # C * T
        compressed_bits = latent_size * quantization_bits
        actual_cr = original_bits / compressed_bits

        # Dequantize (simulate compression/decompression)
        latent_dequantized = dequantize_latent(quantized, metadata)

        # Ensure dtype matches model's expected dtype (float32)
        latent_dequantized = latent_dequantized.float()

        # Decode to reconstruction
        recon = model.decode(latent_dequantized)

        # Ensure output matches input size
        if recon.shape[-1] != noisy.shape[-1]:
            recon = recon[..., :noisy.shape[-1]]

        # Compute metrics per sample
        batch_size = clean.shape[0]
        for i in range(batch_size):
            c = clean[i, 0].cpu().numpy()
            n = noisy[i, 0].cpu().numpy()
            r = recon[i, 0].cpu().numpy()

            # Compute weights for WWPRD
            w = compute_derivative_weights(c, alpha=weight_alpha)

            # Compute metrics
            prd = compute_prd(c, r)
            wwprd = compute_wwprd(c, r, w)
            snr_in = compute_snr(c, n)
            snr_out = compute_snr(c, r)
            snr_imp = snr_out - snr_in

            all_prds.append(prd)
            all_wwprds.append(wwprd)
            all_snr_ins.append(snr_in)
            all_snr_outs.append(snr_out)
            all_snr_improvements.append(snr_imp)
            all_actual_crs.append(actual_cr)

    # Aggregate metrics
    actual_cr_mean = float(np.mean(all_actual_crs))

    metrics = {
        'PRD': float(np.mean(all_prds)),
        'PRD_std': float(np.std(all_prds)),
        'WWPRD': float(np.mean(all_wwprds)),
        'WWPRD_std': float(np.std(all_wwprds)),
        'SNR_in': float(np.mean(all_snr_ins)),
        'SNR_out': float(np.mean(all_snr_outs)),
        'SNR_improvement': float(np.mean(all_snr_improvements)),
        'quantization_bits': quantization_bits,
        'actual_cr': actual_cr_mean,
        'latent_dim': latent.shape[1],  # Channel dimension
    }

    return metrics


def select_quantization_bits(target_cr: float, current_latent_cr: float) -> int:
    """Select appropriate quantization bits to achieve target CR.

    Given the current CR from latent dimension alone, we select quantization
    bits such that: actual_cr ≈ target_cr

    Since: actual_cr = (original_bits) / (latent_size * quantization_bits)
    We need: quantization_bits = (original_bits) / (target_cr * latent_size)

    Args:
        target_cr: Target compression ratio
        current_latent_cr: Current CR from latent dimension only (without quantization)

    Returns:
        Recommended quantization bits (4, 6, or 8)
    """
    # Estimate required quantization bits
    # If current_latent_cr = 0.69, to get CR=4, we need q_bits that makes it 4x larger
    # So q_bits should be current_latent_cr / target_cr * 8 (assuming 8 bits baseline)

    if current_latent_cr > 0:
        estimated_bits = int(current_latent_cr / target_cr * 8)
    else:
        estimated_bits = 8

    # Clamp to valid range and round to nearest valid value
    if target_cr >= 32:
        return 4  # High compression: use 4 bits
    elif target_cr >= 16:
        return 6  # Medium-high compression: use 6 bits
    elif target_cr >= 8:
        return 8  # Medium compression: use 8 bits
    else:
        return 8  # Low compression: use 8 bits (can't go higher easily)


def evaluate_cr_sweep(
    model_path: str,
    config_path: str,
    compression_ratios: List[float],
    quantization_bits: Optional[int] = None,
    num_test_samples: int = 500,
    device: torch.device = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model across multiple compression ratios.

    Args:
        model_path: Path to trained model
        config_path: Path to training config JSON
        compression_ratios: List of target CRs (e.g., [4, 8, 16, 32])
        quantization_bits: Bits for quantization (None = auto-select per CR)
        num_test_samples: Number of test samples to evaluate
        device: Torch device (auto-detect if None)

    Returns:
        Dictionary mapping CR (as string) to metrics dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    console.print(f"[cyan]Using device: {device}")

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load model
    model = load_model(model_path, config, device)

    # Create test dataloader
    test_loader = create_test_dataloader(config, num_test_samples)

    console.print(f"[cyan]Test samples: {len(test_loader.dataset)}")

    # Calculate baseline CR (without quantization) to help select quantization bits
    window_length = int(config.get('window_seconds', 2.0) * config.get('sample_rate', 360))
    original_bits = window_length * 11
    # Get latent shape from model (sample forward pass)
    sample_input = torch.randn(1, 1, window_length).to(device)
    with torch.no_grad():
        sample_latent = model.encode(sample_input)
    latent_size = sample_latent.shape[1] * sample_latent.shape[2]
    baseline_latent_cr = original_bits / (latent_size * 32)  # Assuming 32-bit float

    # Evaluate at each CR
    results = {}

    for cr in compression_ratios:
        # Auto-select quantization bits if not specified
        if quantization_bits is None:
            q_bits = select_quantization_bits(cr, baseline_latent_cr)
        else:
            q_bits = quantization_bits

        metrics = evaluate_at_cr(
            model=model,
            dataloader=test_loader,
            device=device,
            target_cr=cr,
            quantization_bits=q_bits,
            weight_alpha=config.get('weight_alpha', 2.0),
            sample_rate=config.get('sample_rate', 360),
            window_length=int(config.get('window_seconds', 2.0) * config.get('sample_rate', 360)),
        )

        # Store as string key for JSON compatibility (use target CR, not actual CR)
        cr_key = str(int(cr))
        results[cr_key] = metrics

        console.print(f"[green]✓ CR={cr:.0f}:1 (actual={metrics['actual_cr']:.2f}:1)  "
                     f"PRD={metrics['PRD']:.2f}%  WWPRD={metrics['WWPRD']:.2f}%  "
                     f"SNR_imp={metrics['SNR_improvement']:.2f}dB  "
                     f"Q-bits={q_bits}")

    return results


def print_results_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted table of results."""
    table = Table(title="Compression Ratio Sweep Results", show_header=True, header_style="bold magenta")

    table.add_column("CR", style="cyan", justify="center")
    table.add_column("PRD (%)", justify="right", style="green")
    table.add_column("WWPRD (%)", justify="right", style="yellow")
    table.add_column("SNR Imp (dB)", justify="right", style="blue")
    table.add_column("Quant Bits", justify="center", style="white")
    table.add_column("Actual CR", justify="right", style="magenta")

    # Sort by CR value (integer)
    sorted_crs = sorted(results.keys(), key=int)

    for cr_str in sorted_crs:
        cr = int(cr_str)
        m = results[cr_str]
        table.add_row(
            f"{cr}:1",
            f"{m['PRD']:.2f} ± {m['PRD_std']:.2f}",
            f"{m['WWPRD']:.2f} ± {m['WWPRD_std']:.2f}",
            f"{m['SNR_improvement']:.2f}",
            str(m['quantization_bits']),
            f"{m.get('actual_cr', 0):.2f}:1"
        )

    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ECG compression at different compression ratios"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )

    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to training config JSON file'
    )

    parser.add_argument(
        '--compression_ratios',
        type=int,
        nargs='+',
        default=[4, 8, 16, 32],
        help='Compression ratios to evaluate (default: 4 8 16 32)'
    )

    parser.add_argument(
        '--quantization_bits',
        type=int,
        default=8,
        choices=[4, 6, 8],
        help='Quantization bits per latent variable (default: 8)'
    )

    parser.add_argument(
        '--num_test_samples',
        type=int,
        default=500,
        help='Number of test samples to evaluate (default: 500)'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default='outputs/week2/cr_sweep_results.json',
        help='Output JSON file path (default: outputs/week2/cr_sweep_results.json)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    console.print("\n[bold cyan]═══ Week 2: Compression Ratio Evaluation ═══[/bold cyan]\n")

    # Evaluate CR sweep
    results = evaluate_cr_sweep(
        model_path=args.model_path,
        config_path=args.config_path,
        compression_ratios=args.compression_ratios,
        quantization_bits=args.quantization_bits,
        num_test_samples=args.num_test_samples,
        device=device,
    )

    # Print results table
    print_results_table(results)

    # Save to JSON
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]✓ Saved results to {output_path}")
    console.print(f"[cyan]Ready for Person B visualization pipeline!")
    console.print(f"\n[bold]Next step:[/bold]")
    console.print(f"  python -m scripts.plot_rate_distortion \\")
    console.print(f"      --results_file {args.output_file} \\")
    console.print(f"      --output_dir outputs/week2_final\n")


if __name__ == "__main__":
    main()

