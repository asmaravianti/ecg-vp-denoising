"""Plot ablation study results for Week 3.

Creates comparison plots for:
- Loss ablation (MSE vs PRD vs WWPRD)
- Noise ablation (with vs without noise)
- Bottleneck sweep (rate-distortion curves)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console

console = Console()
sns.set_style("whitegrid")


def load_metrics(output_dir: Path) -> Dict:
    """Load final metrics from a training output directory."""
    metrics_file = output_dir / "final_metrics.json"
    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_loss_ablation(results_dir: Path, output_path: Path):
    """Plot loss ablation results."""
    # Find all loss ablation directories
    ablation_dirs = {
        "MSE": None,
        "PRD": None,
        "WWPRD": None
    }

    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            if "mse" in subdir.name.lower():
                ablation_dirs["MSE"] = subdir
            elif "prd" in subdir.name.lower() and "wwprd" not in subdir.name.lower():
                ablation_dirs["PRD"] = subdir
            elif "wwprd" in subdir.name.lower():
                ablation_dirs["WWPRD"] = subdir

    # Load metrics
    metrics = {}
    for loss_name, dir_path in ablation_dirs.items():
        if dir_path:
            metrics[loss_name] = load_metrics(dir_path)

    if not metrics:
        console.print(f"[red]No metrics found in {results_dir}")
        return

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PRD comparison
    ax = axes[0, 0]
    loss_names = list(metrics.keys())
    prd_values = [metrics[name]['PRD'] for name in loss_names if metrics[name]]
    ax.bar(loss_names, prd_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_ylabel('PRD (%)', fontsize=12)
    ax.set_title('PRD Comparison by Loss Function', fontsize=14, fontweight='bold')
    ax.axhline(y=4.33, color='r', linestyle='--', label='Excellent threshold', alpha=0.7)
    ax.axhline(y=9.00, color='orange', linestyle='--', label='Very Good threshold', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # WWPRD comparison
    ax = axes[0, 1]
    wwprd_values = [metrics[name]['WWPRD'] for name in loss_names if metrics[name]]
    ax.bar(loss_names, wwprd_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_ylabel('WWPRD (%)', fontsize=12)
    ax.set_title('WWPRD Comparison by Loss Function', fontsize=14, fontweight='bold')
    ax.axhline(y=7.4, color='r', linestyle='--', label='Excellent threshold', alpha=0.7)
    ax.axhline(y=14.8, color='orange', linestyle='--', label='Very Good threshold', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SNR improvement comparison
    ax = axes[1, 0]
    snr_values = [metrics[name].get('SNR_improvement', 0) for name in loss_names if metrics[name]]
    ax.bar(loss_names, snr_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_ylabel('SNR Improvement (dB)', fontsize=12)
    ax.set_title('SNR Improvement by Loss Function', fontsize=14, fontweight='bold')
    ax.axhline(y=5.0, color='g', linestyle='--', label='Target', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    for name in loss_names:
        if metrics[name]:
            table_data.append([
                name,
                f"{metrics[name]['PRD']:.2f}%",
                f"{metrics[name]['WWPRD']:.2f}%",
                f"{metrics[name].get('SNR_improvement', 0):.2f} dB"
            ])

    table = ax.table(cellText=table_data,
                     colLabels=['Loss', 'PRD', 'WWPRD', 'SNR Imp'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Summary Table', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved loss ablation plot to {output_path}")


def plot_rate_distortion(sweep_dir: Path, output_path: Path):
    """Plot rate-distortion curve from bottleneck sweep."""
    # Load all results
    results = []

    for subdir in sweep_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("latent_"):
            latent_dim = int(subdir.name.split("_")[1])
            metrics = load_metrics(subdir)
            if metrics:
                results.append({
                    'latent_dim': latent_dim,
                    'PRD': metrics.get('PRD', 0),
                    'WWPRD': metrics.get('WWPRD', 0),
                    'CR': metrics.get('CR', 0),
                    'SNR_improvement': metrics.get('SNR_improvement', 0)
                })

    if not results:
        console.print(f"[red]No results found in {sweep_dir}")
        return

    # Sort by latent_dim
    results.sort(key=lambda x: x['latent_dim'])

    # Extract data
    latent_dims = [r['latent_dim'] for r in results]
    prds = [r['PRD'] for r in results]
    wwprds = [r['WWPRD'] for r in results]
    crs = [r['CR'] for r in results]

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PRD vs CR
    ax = axes[0]
    ax.plot(crs, prds, 'o-', linewidth=2, markersize=8, label='PRD')
    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('PRD (%)', fontsize=12)
    ax.set_title('Rate-Distortion Curve (PRD)', fontsize=14, fontweight='bold')
    ax.axhline(y=4.33, color='r', linestyle='--', label='Excellent', alpha=0.7)
    ax.axhline(y=9.00, color='orange', linestyle='--', label='Very Good', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # WWPRD vs CR
    ax = axes[1]
    ax.plot(crs, wwprds, 's-', linewidth=2, markersize=8, label='WWPRD', color='orange')
    ax.set_xlabel('Compression Ratio', fontsize=12)
    ax.set_ylabel('WWPRD (%)', fontsize=12)
    ax.set_title('Rate-Distortion Curve (WWPRD)', fontsize=14, fontweight='bold')
    ax.axhline(y=7.4, color='r', linestyle='--', label='Excellent', alpha=0.7)
    ax.axhline(y=14.8, color='orange', linestyle='--', label='Very Good', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved rate-distortion plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--ablation_type", type=str, required=True,
                        choices=["loss", "noise", "bottleneck"],
                        help="Type of ablation to plot")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing ablation results")
    parser.add_argument("--output", type=str, required=True,
                        help="Output plot file path")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.ablation_type == "loss":
        plot_loss_ablation(results_dir, output_path)
    elif args.ablation_type == "bottleneck":
        plot_rate_distortion(results_dir, output_path)
    elif args.ablation_type == "noise":
        # Similar to loss ablation
        plot_loss_ablation(results_dir, output_path)


if __name__ == "__main__":
    main()

