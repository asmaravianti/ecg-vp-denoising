"""Evaluation script for trained ECG autoencoder.

Generates comprehensive PRD/WWPRD analysis and visualizations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from ecgdae.data import MITBIHDataset, NSTDBNoiseMixer, WindowingConfig, gaussian_snr_mixer
from ecgdae.models import ConvAutoEncoder, ResidualAutoEncoder
from ecgdae.metrics import (
    compute_prd, compute_wwprd, compute_snr,
    compute_derivative_weights, evaluate_reconstruction
)

console = Console()
sns.set_style("whitegrid")


def setup_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ECG autoencoder")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to training config JSON")
    parser.add_argument("--output_dir", type=str, default="./outputs/week1/evaluation",
                        help="Output directory for evaluation results")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use")

    return parser.parse_args()


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
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    console.print(f"[green]✓ Loaded model from {model_path}")
    console.print(f"[green]  Epoch: {checkpoint['epoch']}")
    console.print(f"[green]  Val Loss: {checkpoint['val_loss']:.4f}")

    return model


def create_test_dataloader(config: dict) -> DataLoader:
    """Create test dataloader from config."""
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

    # Use different records for testing (last 5 records)
    from ecgdae.data import MITBIHLoader
    all_records = MITBIHLoader.MITBIH_RECORDS
    test_records = all_records[config['num_records']:config['num_records']+5]

    console.print(f"[yellow]Test records: {test_records}")

    # Create dataset
    dataset = MITBIHDataset(
        records=test_records,
        config=window_config,
        noise_mixer=noise_mixer,
        data_dir=config.get('data_dir', './data/mitbih'),
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
def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect predictions from model.

    Returns:
        clean, noisy, reconstructed arrays
    """
    model.eval()

    all_clean = []
    all_noisy = []
    all_recon = []

    total_collected = 0

    for noisy, clean in dataloader:
        if total_collected >= num_samples:
            break

        noisy = noisy.to(device)
        clean = clean.to(device)

        recon = model(noisy)

        all_clean.append(clean.cpu())
        all_noisy.append(noisy.cpu())
        all_recon.append(recon.cpu())

        total_collected += clean.shape[0]

    # Concatenate
    all_clean = torch.cat(all_clean, dim=0)[:num_samples]
    all_noisy = torch.cat(all_noisy, dim=0)[:num_samples]
    all_recon = torch.cat(all_recon, dim=0)[:num_samples]

    return all_clean.numpy(), all_noisy.numpy(), all_recon.numpy()


def compute_per_sample_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    recon: np.ndarray,
    weight_alpha: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Compute metrics for each sample.

    Args:
        clean: (N, C, T)
        noisy: (N, C, T)
        recon: (N, C, T)

    Returns:
        Dictionary of metric arrays
    """
    num_samples = clean.shape[0]

    prds = []
    wwprds = []
    snr_ins = []
    snr_outs = []
    snr_improvements = []

    console.print("[yellow]Computing per-sample metrics...")

    for i in range(num_samples):
        c = clean[i]
        n = noisy[i]
        r = recon[i]

        # Compute weights for WWPRD
        w = compute_derivative_weights(c, alpha=weight_alpha)

        # Compute metrics
        prd = compute_prd(c, r)
        wwprd = compute_wwprd(c, r, w)
        snr_in = compute_snr(c, n)
        snr_out = compute_snr(c, r)
        snr_imp = snr_out - snr_in

        prds.append(prd)
        wwprds.append(wwprd)
        snr_ins.append(snr_in)
        snr_outs.append(snr_out)
        snr_improvements.append(snr_imp)

    return {
        'PRD': np.array(prds),
        'WWPRD': np.array(wwprds),
        'SNR_in': np.array(snr_ins),
        'SNR_out': np.array(snr_outs),
        'SNR_improvement': np.array(snr_improvements),
    }


def plot_metric_distributions(metrics: Dict[str, np.ndarray], output_dir: Path):
    """Plot distributions of PRD and WWPRD."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PRD histogram
    ax = axes[0, 0]
    ax.hist(metrics['PRD'], bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(metrics['PRD']), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(metrics["PRD"]):.2f}%')
    ax.axvline(4.33, color='green', linestyle=':', linewidth=2, label='Excellent threshold')
    ax.axvline(9.00, color='orange', linestyle=':', linewidth=2, label='Very Good threshold')
    ax.set_xlabel('PRD (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('PRD Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # WWPRD histogram
    ax = axes[0, 1]
    ax.hist(metrics['WWPRD'], bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(metrics['WWPRD']), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(metrics["WWPRD"]):.2f}%')
    ax.axvline(7.4, color='green', linestyle=':', linewidth=2, label='Excellent threshold')
    ax.axvline(14.8, color='orange', linestyle=':', linewidth=2, label='Very Good threshold')
    ax.set_xlabel('WWPRD (%)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('WWPRD Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # SNR comparison
    ax = axes[1, 0]
    ax.hist(metrics['SNR_in'], bins=30, alpha=0.5, label='Input SNR', color='gray')
    ax.hist(metrics['SNR_out'], bins=30, alpha=0.5, label='Output SNR', color='blue')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('SNR Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # SNR improvement
    ax = axes[1, 1]
    ax.hist(metrics['SNR_improvement'], bins=40, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(metrics['SNR_improvement']), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(metrics["SNR_improvement"]):.2f} dB')
    ax.set_xlabel('SNR Improvement (dB)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('SNR Improvement Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metric_distributions.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved metric distributions to {output_dir / 'metric_distributions.png'}")
    plt.close()


def plot_prd_wwprd_scatter(metrics: Dict[str, np.ndarray], output_dir: Path):
    """Plot PRD vs WWPRD scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    scatter = ax.scatter(metrics['PRD'], metrics['WWPRD'],
                        c=metrics['SNR_improvement'], cmap='viridis',
                        s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('SNR Improvement (dB)', fontsize=12)

    # Add threshold lines
    ax.axvline(4.33, color='green', linestyle='--', alpha=0.5, label='PRD Excellent')
    ax.axvline(9.00, color='orange', linestyle='--', alpha=0.5, label='PRD Very Good')
    ax.axhline(7.4, color='green', linestyle=':', alpha=0.5, label='WWPRD Excellent')
    ax.axhline(14.8, color='orange', linestyle=':', alpha=0.5, label='WWPRD Very Good')

    # Add diagonal reference (PRD = WWPRD)
    max_val = max(np.max(metrics['PRD']), np.max(metrics['WWPRD']))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.3, linewidth=1, label='PRD = WWPRD')

    ax.set_xlabel('PRD (%)', fontsize=13)
    ax.set_ylabel('WWPRD (%)', fontsize=13)
    ax.set_title('PRD vs WWPRD Scatter Plot', fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "prd_wwprd_scatter.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved PRD-WWPRD scatter plot to {output_dir / 'prd_wwprd_scatter.png'}")
    plt.close()


def plot_quality_classification(metrics: Dict[str, np.ndarray], output_dir: Path):
    """Plot quality classification pie charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PRD quality classification
    prd_excellent = np.sum(metrics['PRD'] < 4.33)
    prd_very_good = np.sum((metrics['PRD'] >= 4.33) & (metrics['PRD'] < 9.00))
    prd_good = np.sum((metrics['PRD'] >= 9.00) & (metrics['PRD'] < 15.00))
    prd_not_good = np.sum(metrics['PRD'] >= 15.00)

    prd_counts = [prd_excellent, prd_very_good, prd_good, prd_not_good]
    prd_labels = ['Excellent\n(<4.33%)', 'Very Good\n(4.33-9%)',
                  'Good\n(9-15%)', 'Not Good\n(≥15%)']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

    ax = axes[0]
    wedges, texts, autotexts = ax.pie(prd_counts, labels=prd_labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('PRD Quality Classification', fontsize=14, fontweight='bold', pad=20)

    # WWPRD quality classification
    wwprd_excellent = np.sum(metrics['WWPRD'] < 7.4)
    wwprd_very_good = np.sum((metrics['WWPRD'] >= 7.4) & (metrics['WWPRD'] < 14.8))
    wwprd_good = np.sum((metrics['WWPRD'] >= 14.8) & (metrics['WWPRD'] < 24.7))
    wwprd_not_good = np.sum(metrics['WWPRD'] >= 24.7)

    wwprd_counts = [wwprd_excellent, wwprd_very_good, wwprd_good, wwprd_not_good]
    wwprd_labels = ['Excellent\n(<7.4%)', 'Very Good\n(7.4-14.8%)',
                    'Good\n(14.8-24.7%)', 'Not Good\n(≥24.7%)']

    ax = axes[1]
    wedges, texts, autotexts = ax.pie(wwprd_counts, labels=wwprd_labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('WWPRD Quality Classification', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "quality_classification.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved quality classification to {output_dir / 'quality_classification.png'}")
    plt.close()


def plot_reconstruction_gallery(
    clean: np.ndarray,
    noisy: np.ndarray,
    recon: np.ndarray,
    metrics: Dict[str, np.ndarray],
    output_dir: Path,
    num_examples: int = 8,
):
    """Plot a gallery of reconstruction examples."""
    # Select best and worst examples
    sorted_indices = np.argsort(metrics['PRD'])
    best_indices = sorted_indices[:num_examples // 2]
    worst_indices = sorted_indices[-(num_examples // 2):]
    selected_indices = np.concatenate([best_indices, worst_indices])

    fig, axes = plt.subplots(num_examples, 1, figsize=(16, 2.5 * num_examples))

    for i, idx in enumerate(selected_indices):
        ax = axes[i]

        time = np.arange(clean.shape[-1]) / 360.0

        ax.plot(time, clean[idx, 0], 'g-', label='Clean', linewidth=1.5, alpha=0.8)
        ax.plot(time, noisy[idx, 0], color='gray', label='Noisy', linewidth=1, alpha=0.5)
        ax.plot(time, recon[idx, 0], 'r--', label='Reconstructed', linewidth=1.5, alpha=0.8)

        # Add metrics to title
        prd = metrics['PRD'][idx]
        wwprd = metrics['WWPRD'][idx]
        snr_imp = metrics['SNR_improvement'][idx]

        quality = 'Best' if i < num_examples // 2 else 'Worst'

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'{quality} Example - PRD: {prd:.2f}%, WWPRD: {wwprd:.2f}%, SNR Imp: {snr_imp:.2f} dB',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_gallery.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]✓ Saved reconstruction gallery to {output_dir / 'reconstruction_gallery.png'}")
    plt.close()


def print_summary_table(metrics: Dict[str, np.ndarray]):
    """Print a summary table of metrics."""
    table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")

    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std", justify="right", style="yellow")
    table.add_column("Min", justify="right", style="blue")
    table.add_column("Max", justify="right", style="red")

    for metric_name, values in metrics.items():
        table.add_row(
            metric_name,
            f"{np.mean(values):.2f}",
            f"{np.std(values):.2f}",
            f"{np.min(values):.2f}",
            f"{np.max(values):.2f}",
        )

    console.print("\n")
    console.print(table)
    console.print("\n")


def main():
    args = setup_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    console.print(f"[bold cyan]Using device: {device}\n")

    # Load config
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    console.print(f"[cyan]Loaded config from {args.config_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path, config, device)

    # Create test dataloader
    test_loader = create_test_dataloader(config)

    # Collect predictions
    console.print(f"[yellow]Collecting predictions for {args.num_samples} samples...")
    clean, noisy, recon = collect_predictions(model, test_loader, device, args.num_samples)

    console.print(f"[green]✓ Collected {clean.shape[0]} samples")

    # Compute per-sample metrics
    metrics = compute_per_sample_metrics(
        clean, noisy, recon,
        weight_alpha=config.get('weight_alpha', 2.0)
    )

    # Print summary
    print_summary_table(metrics)

    # Save metrics to JSON
    metrics_json = {k: v.tolist() for k, v in metrics.items()}
    with open(output_dir / "evaluation_metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)

    console.print(f"[green]✓ Saved metrics to {output_dir / 'evaluation_metrics.json'}")

    # Generate plots
    console.print("[yellow]Generating visualizations...")
    plot_metric_distributions(metrics, output_dir)
    plot_prd_wwprd_scatter(metrics, output_dir)
    plot_quality_classification(metrics, output_dir)
    plot_reconstruction_gallery(clean, noisy, recon, metrics, output_dir)

    console.print(f"\n[bold green]✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

