"""Calculate Quality Scores (QS, QSN) and generate comparison table.

This script:
1. Evaluates models at different compression ratios
2. Calculates QS = CR/PRD and QSN = CR/PRDN
3. Generates a table similar to Table IV from the professor's paper
4. Compares WWPRD-only vs Combined loss methods

Author: Based on professor's paper requirements
Date: 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ecgdae.data import MITBIHDataset, WindowingConfig, NSTDBNoiseMixer, gaussian_snr_mixer
from ecgdae.models import ConvAutoEncoder, ResidualAutoEncoder
from ecgdae.metrics import compute_prd, compute_prdn, compute_wwprd_wavelet
from ecgdae.quantization import quantize_latent, dequantize_latent, compute_compression_ratio
from torch.utils.data import DataLoader

console = Console()


def load_model(model_path: str, config_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model architecture
    if config['model_type'] == 'residual':
        model = ResidualAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(config['hidden_dims']),
            latent_dim=config['latent_dim'],
        )
    else:
        model = ConvAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(config['hidden_dims']),
            latent_dim=config['latent_dim'],
        )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model, config


def evaluate_model_on_records(
    model: torch.nn.Module,
    config: dict,
    record_ids: List[int],
    quantization_bits: int = 8,
    device: torch.device = None,
) -> Dict[int, Dict[str, float]]:
    """Evaluate model on specific MIT-BIH records.
    
    Returns:
        Dictionary mapping record_id to metrics dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # Create windowing config
    window_config = WindowingConfig(
        sample_rate=config['sample_rate'],
        window_seconds=config['window_seconds'],
        step_seconds=config['window_seconds'],  # Non-overlapping windows
    )
    
    # Create noise mixer
    if config.get('noise_type') == 'nstdb':
        nstdb = NSTDBNoiseMixer(data_dir=config.get('nstdb_dir', './data/nstdb'))
        noise_mixer = nstdb.create_mixer(
            target_snr_db=config.get('snr_db', 10.0),
            noise_type=config.get('nstdb_noise', 'muscle_artifact'),
            mix_gaussian=False,
        )
    else:
        # Gaussian noise
        noise_mixer = gaussian_snr_mixer(config.get('snr_db', 10.0))
    
    for record_id in record_ids:
        console.print(f"[cyan]Evaluating record {record_id}...[/cyan]")
        
        # Convert record_id to string format (e.g., 117 -> "117")
        record_name = str(record_id)
        
        # Load data for this record
        dataset = MITBIHDataset(
            records=[record_name],
            config=window_config,
            noise_mixer=noise_mixer,
            data_dir=config['data_dir'],
            channel=0,
            normalize=True,
        )
        
        if len(dataset) == 0:
            console.print(f"[yellow]⚠ Record {record_id} not found, skipping...[/yellow]")
            continue
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Collect metrics across all windows in this record
        all_prds = []
        all_prdns = []
        all_wwprds = []
        all_crs = []
        
        with torch.no_grad():
            for noisy, clean in dataloader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                # Encode
                latent = model.encode(noisy)
                
                # Quantize
                quantized, metadata = quantize_latent(latent, quantization_bits, return_metadata=True)
                
                # Calculate CR
                window_length = clean.shape[-1]
                original_bits = window_length * 11  # 11 bits per sample
                latent_shape = latent.shape  # (B, C, T)
                latent_size = latent_shape[1] * latent_shape[2]
                compressed_bits = latent_size * quantization_bits
                cr = original_bits / compressed_bits
                
                # Dequantize and decode
                latent_dequantized = dequantize_latent(quantized, metadata).float()
                recon = model.decode(latent_dequantized)
                
                # Ensure output matches input size
                if recon.shape[-1] != clean.shape[-1]:
                    recon = recon[..., :clean.shape[-1]]
                
                # Compute metrics per sample
                batch_size = clean.shape[0]
                for i in range(batch_size):
                    c = clean[i, 0].cpu().numpy()
                    r = recon[i, 0].cpu().numpy()
                    
                    prd = compute_prd(c, r)
                    prdn = compute_prdn(c, r)
                    
                    try:
                        wwprd = compute_wwprd_wavelet(c, r)
                    except:
                        wwprd = float('nan')
                    
                    all_prds.append(prd)
                    all_prdns.append(prdn)
                    all_wwprds.append(wwprd)
                    all_crs.append(cr)
        
        # Average metrics for this record
        if all_prds:
            results[record_id] = {
                'PRDN': np.mean(all_prdns),
                'PRDN_std': np.std(all_prdns),
                'WWPRD': np.nanmean(all_wwprds) if not np.isnan(all_wwprds).all() else float('nan'),
                'WWPRD_std': np.nanstd(all_wwprds) if not np.isnan(all_wwprds).all() else float('nan'),
                'CR': np.mean(all_crs),
                'PRD': np.mean(all_prds),  # Also compute PRD for QS calculation
            }
            
            # Calculate QS and QSN
            results[record_id]['QS'] = results[record_id]['CR'] / (results[record_id]['PRD'] + 1e-6)
            results[record_id]['QSN'] = results[record_id]['CR'] / (results[record_id]['PRDN'] + 1e-6)
        else:
            console.print(f"[red]✗ No data for record {record_id}[/red]")
    
    return results


def generate_comparison_table(
    wwprd_results: Dict[int, Dict[str, float]],
    combined_results: Dict[int, Dict[str, float]],
    output_path: str = None,
) -> str:
    """Generate a table similar to Table IV from the professor's paper.
    
    Args:
        wwprd_results: Results from WWPRD-only model
        combined_results: Results from Combined loss model
        output_path: Optional path to save table as CSV/JSON
    
    Returns:
        Formatted table string
    """
    # Get all record IDs (union of both result sets)
    all_records = sorted(set(list(wwprd_results.keys()) + list(combined_results.keys())))
    
    # Create table
    table = Table(title="Compression Results: WWPRD vs Combined Loss", show_header=True, header_style="bold magenta")
    
    table.add_column("Record", style="cyan", no_wrap=True)
    table.add_column("Method", style="yellow")
    table.add_column("PRDN (%)", justify="right", style="green")
    table.add_column("WWPRD (%)", justify="right", style="blue")
    table.add_column("CR 1:X", justify="right", style="magenta")
    table.add_column("QSN", justify="right", style="yellow")
    table.add_column("QS", justify="right", style="yellow")
    
    # Add rows for each record
    for record_id in all_records:
        # WWPRD-only row
        if record_id in wwprd_results:
            r = wwprd_results[record_id]
            table.add_row(
                str(record_id),
                "WWPRD-only",
                f"{r['PRDN']:.2f}",
                f"{r['WWPRD']:.2f}" if not np.isnan(r['WWPRD']) else "N/A",
                f"{r['CR']:.2f}",
                f"{r['QSN']:.3f}",
                f"{r['QS']:.3f}",
            )
        
        # Combined loss row
        if record_id in combined_results:
            r = combined_results[record_id]
            table.add_row(
                str(record_id),
                "Combined",
                f"{r['PRDN']:.2f}",
                f"{r['WWPRD']:.2f}" if not np.isnan(r['WWPRD']) else "N/A",
                f"{r['CR']:.2f}",
                f"{r['QSN']:.3f}",
                f"{r['QS']:.3f}",
            )
    
    # Calculate averages
    def calc_avg(results_dict: Dict[int, Dict[str, float]], key: str) -> Tuple[float, float]:
        values = [r[key] for r in results_dict.values() if not np.isnan(r.get(key, np.nan))]
        if values:
            return np.mean(values), np.std(values)
        return float('nan'), float('nan')
    
    # Average row for WWPRD-only
    wwprd_prdn_avg, wwprd_prdn_std = calc_avg(wwprd_results, 'PRDN')
    wwprd_wwprd_avg, wwprd_wwprd_std = calc_avg(wwprd_results, 'WWPRD')
    wwprd_cr_avg, wwprd_cr_std = calc_avg(wwprd_results, 'CR')
    wwprd_qsn_avg, _ = calc_avg(wwprd_results, 'QSN')
    wwprd_qs_avg, _ = calc_avg(wwprd_results, 'QS')
    
    table.add_section()
    table.add_row(
        "Average",
        "WWPRD-only",
        f"{wwprd_prdn_avg:.2f} ± {wwprd_prdn_std:.2f}",
        f"{wwprd_wwprd_avg:.2f} ± {wwprd_wwprd_std:.2f}" if not np.isnan(wwprd_wwprd_avg) else "N/A",
        f"{wwprd_cr_avg:.2f} ± {wwprd_cr_std:.2f}",
        f"{wwprd_qsn_avg:.3f}",
        f"{wwprd_qs_avg:.3f}",
        style="bold",
    )
    
    # Average row for Combined
    comb_prdn_avg, comb_prdn_std = calc_avg(combined_results, 'PRDN')
    comb_wwprd_avg, comb_wwprd_std = calc_avg(combined_results, 'WWPRD')
    comb_cr_avg, comb_cr_std = calc_avg(combined_results, 'CR')
    comb_qsn_avg, _ = calc_avg(combined_results, 'QSN')
    comb_qs_avg, _ = calc_avg(combined_results, 'QS')
    
    table.add_row(
        "Average",
        "Combined",
        f"{comb_prdn_avg:.2f} ± {comb_prdn_std:.2f}",
        f"{comb_wwprd_avg:.2f} ± {comb_wwprd_std:.2f}" if not np.isnan(comb_wwprd_avg) else "N/A",
        f"{comb_cr_avg:.2f} ± {comb_cr_std:.2f}",
        f"{comb_qsn_avg:.3f}",
        f"{comb_qs_avg:.3f}",
        style="bold",
    )
    
    # Display table
    console.print("\n")
    console.print(table)
    
    # Save to file if requested
    if output_path:
        # Save as JSON for further processing
        json_path = Path(output_path).with_suffix('.json')
        output_data = {
            'wwprd_results': wwprd_results,
            'combined_results': combined_results,
            'averages': {
                'wwprd': {
                    'PRDN_avg': float(wwprd_prdn_avg),
                    'PRDN_std': float(wwprd_prdn_std),
                    'WWPRD_avg': float(wwprd_wwprd_avg) if not np.isnan(wwprd_wwprd_avg) else None,
                    'WWPRD_std': float(wwprd_wwprd_std) if not np.isnan(wwprd_wwprd_std) else None,
                    'CR_avg': float(wwprd_cr_avg),
                    'CR_std': float(wwprd_cr_std),
                    'QSN_avg': float(wwprd_qsn_avg),
                    'QS_avg': float(wwprd_qs_avg),
                },
                'combined': {
                    'PRDN_avg': float(comb_prdn_avg),
                    'PRDN_std': float(comb_prdn_std),
                    'WWPRD_avg': float(comb_wwprd_avg) if not np.isnan(comb_wwprd_avg) else None,
                    'WWPRD_std': float(comb_wwprd_std) if not np.isnan(comb_wwprd_std) else None,
                    'CR_avg': float(comb_cr_avg),
                    'CR_std': float(comb_cr_std),
                    'QSN_avg': float(comb_qsn_avg),
                    'QS_avg': float(comb_qs_avg),
                },
            },
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"\n[green]✓ Results saved to {json_path}[/green]")
    
    return str(table)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate QS scores and generate comparison table"
    )
    parser.add_argument(
        '--wwprd_model',
        type=str,
        required=True,
        help='Path to WWPRD-only model checkpoint'
    )
    parser.add_argument(
        '--wwprd_config',
        type=str,
        required=True,
        help='Path to WWPRD-only model config JSON'
    )
    parser.add_argument(
        '--combined_model',
        type=str,
        required=True,
        help='Path to Combined loss model checkpoint'
    )
    parser.add_argument(
        '--combined_config',
        type=str,
        required=True,
        help='Path to Combined loss model config JSON'
    )
    parser.add_argument(
        '--record_ids',
        type=int,
        nargs='+',
        default=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232],
        help='MIT-BIH record IDs to evaluate (default: all common records)'
    )
    parser.add_argument(
        '--quantization_bits',
        type=int,
        default=8,
        help='Quantization bits for compression (default: 8)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/comparison_table.json',
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    console.print(f"[bold green]Quality Score (QS) Calculation and Comparison Table[/bold green]")
    console.print(f"[cyan]Device: {device}[/cyan]\n")
    
    # Load models
    console.print("[bold]Step 1: Loading models...[/bold]")
    wwprd_model, wwprd_config = load_model(args.wwprd_model, args.wwprd_config, device)
    console.print(f"[green]✓ Loaded WWPRD-only model[/green]")
    
    combined_model, combined_config = load_model(args.combined_model, args.combined_config, device)
    console.print(f"[green]✓ Loaded Combined loss model[/green]\n")
    
    # Evaluate WWPRD-only model
    console.print("[bold]Step 2: Evaluating WWPRD-only model...[/bold]")
    wwprd_results = evaluate_model_on_records(
        wwprd_model,
        wwprd_config,
        args.record_ids,
        quantization_bits=args.quantization_bits,
        device=device,
    )
    console.print(f"[green]✓ Evaluated {len(wwprd_results)} records[/green]\n")
    
    # Evaluate Combined loss model
    console.print("[bold]Step 3: Evaluating Combined loss model...[/bold]")
    combined_results = evaluate_model_on_records(
        combined_model,
        combined_config,
        args.record_ids,
        quantization_bits=args.quantization_bits,
        device=device,
    )
    console.print(f"[green]✓ Evaluated {len(combined_results)} records[/green]\n")
    
    # Generate comparison table
    console.print("[bold]Step 4: Generating comparison table...[/bold]")
    generate_comparison_table(wwprd_results, combined_results, output_path=args.output)
    
    console.print("\n[bold green]✓ Analysis complete![/bold green]")


if __name__ == '__main__':
    main()

