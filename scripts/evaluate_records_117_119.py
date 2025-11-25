"""Evaluate model on MIT-BIH records 117 and 119 for comparison with paper Tab IV.

This script:
1. Loads the trained model
2. Evaluates on records 117 and 119 from MIT-BIH Arrhythmia dataset
3. Computes PRDN and WWPRD metrics (using correct formulas)
4. Outputs results in a format suitable for comparison with paper Tab IV
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from ecgdae.data import MITBIHDataset, MITBIHLoader, WindowingConfig, NSTDBNoiseMixer, gaussian_snr_mixer
from ecgdae.metrics import compute_prdn, compute_wwprd_wavelet, compute_wwprd, compute_derivative_weights
from ecgdae.models import ConvAutoEncoder, ResidualAutoEncoder

console = Console()


def load_model(model_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Determine model type
    model_type = config.get('model_type', 'conv')
    hidden_dims = config.get('hidden_dims', [32, 64, 128])
    latent_dim = config.get('latent_dim', 32)

    # Calculate window size
    window_seconds = config.get('window_seconds', 2.0)
    sample_rate = config.get('sample_rate', 360)
    window_size = int(window_seconds * sample_rate)

    # Create model
    if model_type == 'conv':
        model = ConvAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(hidden_dims),
            latent_dim=latent_dim,
        )
    elif model_type == 'residual':
        model = ResidualAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(hidden_dims),
            latent_dim=latent_dim,
            num_res_blocks=2,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    console.print(f"[green]✓ Loaded {model_type} model from {model_path}")
    return model


def create_record_dataloader(
    record_names: List[str],
    config: dict,
    data_dir: str = "./data/mitbih",
) -> DataLoader:
    """Create dataloader for specific MIT-BIH records."""
    # Setup windowing
    window_config = WindowingConfig(
        sample_rate=config['sample_rate'],
        window_seconds=config['window_seconds'],
        step_seconds=config['window_seconds'],  # Non-overlapping windows
    )

    # Setup noise mixer (same as training)
    if config['noise_type'] == 'gaussian':
        noise_mixer = gaussian_snr_mixer(config['snr_db'])
    else:
        nstdb = NSTDBNoiseMixer(data_dir="./data/nstdb")
        noise_mixer = nstdb.create_mixer(
            target_snr_db=config['snr_db'],
            noise_type=config['nstdb_noise'],
        )

    console.print(f"[cyan]Loading records: {record_names}")

    # Create dataset
    dataset = MITBIHDataset(
        records=record_names,
        config=window_config,
        noise_mixer=noise_mixer,
        data_dir=data_dir,
        channel=0,
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    return loader


@torch.no_grad()
def evaluate_records(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    weight_alpha: float = 2.0,
) -> Dict[str, float]:
    """Evaluate model on records and compute PRDN and WWPRD."""
    model.eval()

    all_prdns = []
    all_wwprds_wavelet = []
    all_wwprds_deriv = []

    for noisy_batch, clean_batch in dataloader:
        noisy_batch = noisy_batch.to(device)
        clean_batch = clean_batch.to(device)

        # Reconstruct
        recon_batch = model(noisy_batch)

        # Compute metrics for each sample
        batch_size = clean_batch.shape[0]
        for i in range(batch_size):
            clean = clean_batch[i, 0].cpu().numpy()
            recon = recon_batch[i, 0].cpu().numpy()

            # Compute PRDN (normalized PRD with mean removed)
            prdn = compute_prdn(clean, recon)
            all_prdns.append(prdn)

            # Compute WWPRD (wavelet-based, per paper)
            try:
                wwprd_wavelet = compute_wwprd_wavelet(clean, recon)
                all_wwprds_wavelet.append(wwprd_wavelet)
            except Exception as e:
                console.print(f"[yellow]Warning: WWPRD wavelet computation failed: {e}")
                all_wwprds_wavelet.append(float('nan'))

            # Compute WWPRD (derivative-based, for reference)
            weights = compute_derivative_weights(clean, alpha=weight_alpha)
            wwprd_deriv = compute_wwprd(clean, recon, weights)
            all_wwprds_deriv.append(wwprd_deriv)

    # Filter out NaN values for wavelet WWPRD
    valid_wwprds_wavelet = [x for x in all_wwprds_wavelet if not np.isnan(x)]

    metrics = {
        'PRDN_mean': np.mean(all_prdns),
        'PRDN_std': np.std(all_prdns),
        'PRDN_min': np.min(all_prdns),
        'PRDN_max': np.max(all_prdns),
        'WWPRD_mean': np.nanmean(all_wwprds_wavelet) if len(valid_wwprds_wavelet) > 0 else float('nan'),
        'WWPRD_std': np.nanstd(all_wwprds_wavelet) if len(valid_wwprds_wavelet) > 0 else float('nan'),
        'WWPRD_deriv_mean': np.mean(all_wwprds_deriv),
        'WWPRD_deriv_std': np.std(all_wwprds_deriv),
        'num_samples': len(all_prdns),
    }

    return metrics


def print_comparison_table(
    results_117: Dict[str, float],
    results_119: Dict[str, float],
    paper_values: Dict[str, Dict[str, float]] = None,
):
    """Print comparison table with paper values."""
    table = Table(
        title="Evaluation Results: Records 117 & 119 (Comparison with Paper Tab IV)",
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Record 117", justify="right", style="green")
    table.add_column("Record 119", justify="right", style="green")

    if paper_values:
        table.add_column("Paper 117", justify="right", style="yellow")
        table.add_column("Paper 119", justify="right", style="yellow")

    # PRDN
    prdn_117 = f"{results_117['PRDN_mean']:.2f} ± {results_117['PRDN_std']:.2f}"
    prdn_119 = f"{results_119['PRDN_mean']:.2f} ± {results_119['PRDN_std']:.2f}"
    paper_prdn_117 = f"{paper_values['117']['PRDN']:.2f}" if paper_values and '117' in paper_values else "N/A"
    paper_prdn_119 = f"{paper_values['119']['PRDN']:.2f}" if paper_values and '119' in paper_values else "N/A"

    table.add_row(
        "PRDN (mean ± std)",
        prdn_117,
        prdn_119,
        paper_prdn_117,
        paper_prdn_119,
    )

    # WWPRD (wavelet-based)
    if not np.isnan(results_117['WWPRD_mean']):
        wwprd_117 = f"{results_117['WWPRD_mean']:.2f} ± {results_117['WWPRD_std']:.2f}"
    else:
        wwprd_117 = "N/A"

    if not np.isnan(results_119['WWPRD_mean']):
        wwprd_119 = f"{results_119['WWPRD_mean']:.2f} ± {results_119['WWPRD_std']:.2f}"
    else:
        wwprd_119 = "N/A"

    paper_wwprd_117 = f"{paper_values['117']['WWPRD']:.2f}" if paper_values and '117' in paper_values else "N/A"
    paper_wwprd_119 = f"{paper_values['119']['WWPRD']:.2f}" if paper_values and '119' in paper_values else "N/A"

    table.add_row(
        "WWPRD (wavelet, mean ± std)",
        wwprd_117,
        wwprd_119,
        paper_wwprd_117,
        paper_wwprd_119,
    )

    # Sample counts
    table.add_row(
        "Num Samples",
        f"{int(results_117['num_samples'])}",
        f"{int(results_119['num_samples'])}",
        "N/A",
        "N/A",
    )

    console.print("\n")
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on records 117 and 119")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to config.json (default: same dir as model)")
    parser.add_argument("--data_dir", type=str, default="./data/mitbih",
                        help="MIT-BIH data directory")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use")
    parser.add_argument("--paper_values", type=str, default=None,
                        help="Path to JSON file with paper Tab IV values (optional)")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    console.print(f"[bold cyan]Using device: {device}")

    # Load config
    model_path = Path(args.model_path)
    if args.config_path:
        config_path = Path(args.config_path)
    else:
        config_path = model_path.parent / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    console.print(f"[cyan]Config loaded from {config_path}")
    console.print(f"[cyan]Model type: {config.get('model_type', 'conv')}")
    console.print(f"[cyan]Latent dim: {config.get('latent_dim', 32)}")
    console.print(f"[cyan]Loss type: {config.get('loss_type', 'wwprd')}")

    # Load paper values if provided
    paper_values = None
    if args.paper_values:
        with open(args.paper_values, 'r') as f:
            paper_values = json.load(f)
        console.print(f"[cyan]Paper values loaded from {args.paper_values}")

    # Load model
    model = load_model(str(model_path), config, device)

    # Evaluate on records 117 and 119
    records_to_test = ['117', '119']
    results = {}

    for record_name in records_to_test:
        console.print(f"\n[bold yellow]Evaluating Record {record_name}...")

        # Create dataloader for this record
        dataloader = create_record_dataloader(
            record_names=[record_name],
            config=config,
            data_dir=args.data_dir,
        )

        # Evaluate
        metrics = evaluate_records(
            model=model,
            dataloader=dataloader,
            device=device,
            weight_alpha=config.get('weight_alpha', 2.0),
        )

        results[record_name] = metrics

        console.print(f"[green]Record {record_name} Results:")
        console.print(f"  PRDN: {metrics['PRDN_mean']:.2f} ± {metrics['PRDN_std']:.2f}%")
        if not np.isnan(metrics['WWPRD_mean']):
            console.print(f"  WWPRD (wavelet): {metrics['WWPRD_mean']:.2f} ± {metrics['WWPRD_std']:.2f}%")
        console.print(f"  WWPRD (deriv): {metrics['WWPRD_deriv_mean']:.2f} ± {metrics['WWPRD_deriv_std']:.2f}%")
        console.print(f"  Samples: {int(metrics['num_samples'])}")

    # Print comparison table
    print_comparison_table(results['117'], results['119'], paper_values)

    # Save results
    output_path = model_path.parent / "evaluation_records_117_119.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓ Results saved to {output_path}")
    console.print("\n[bold yellow]Note: Compare these values with Tab IV in the paper.")
    console.print("[yellow]If values are significantly different, check:")
    console.print("  1. PRDN formula uses mean-centered denominator")
    console.print("  2. Same noise type and SNR as paper")
    console.print("  3. Same window size and preprocessing")


if __name__ == "__main__":
    main()

