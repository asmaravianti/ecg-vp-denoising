"""Ablation study training script for Week 3.

Supports:
- Loss ablation: MSE vs PRD vs WWPRD
- Noise ablation: with vs without noise augmentation
- Fixed configuration for fair comparison
"""

import argparse
import json
from pathlib import Path

import sys
from pathlib import Path
# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_mitbih import (
    setup_args, create_model, create_loss_function, create_dataloader,
    train_epoch, validate, plot_training_curves, plot_reconstruction_examples,
    format_metrics, compute_wwprd_weights
)
from ecgdae.models import ConvAutoEncoder, ResidualAutoEncoder
from ecgdae.data import MITBIHDataset, NSTDBNoiseMixer, WindowingConfig, gaussian_snr_mixer
from ecgdae.losses import PRDLoss, WWPRDLoss
from ecgdae.metrics import batch_evaluate
from ecgdae.quantization import compute_compression_ratio

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import track
import numpy as np

console = Console()


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Ablation study training")

    # Inherit all arguments from train_mitbih
    args = setup_args()

    # Override for ablation study
    parser.add_argument("--ablation_type", type=str, default="loss",
                        choices=["loss", "noise"],
                        help="Type of ablation: loss or noise")
    parser.add_argument("--no_noise", action="store_true",
                        help="Train without noise (for noise ablation)")

    # Parse ablation-specific args
    ablation_args, remaining = parser.parse_known_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    console.print(f"[bold cyan]Using device: {device}")
    console.print(f"[cyan]Ablation type: {ablation_args.ablation_type}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = vars(args)
    config['ablation_type'] = ablation_args.ablation_type
    config['no_noise'] = ablation_args.no_noise

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    # Note: For noise ablation with no noise, we need to modify create_dataloader
    # For now, we'll use a workaround by setting noise_type to gaussian with very high SNR
    if ablation_args.no_noise:
        # For noise ablation: no noise (use very high SNR to simulate no noise)
        original_noise_type = args.noise_type
        args.noise_type = "gaussian"
        args.snr_db = 100.0  # Very high SNR ≈ no noise
    else:
        original_noise_type = None

    train_loader, val_loader = create_dataloader(args)

    # Create model
    model = create_model(args).to(device)

    # Create loss function
    loss_fn = create_loss_function(args).to(device)
    console.print(f"[cyan]Loss function: {args.loss_type}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'PRD': [],
        'PRD_vs_noisy': [],
        'WWPRD': [],
        'WWPRD_vs_noisy': [],
        'SNR_out': [],
        'SNR_in': [],
        'SNR_improvement': [],
    }

    # Training loop
    console.print("\n[bold green]Starting ablation training...")

    use_weights = (args.loss_type == "wwprd")
    best_val_loss = float('inf')

    for epoch in track(range(1, args.epochs + 1), description="Training"):
        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            use_weights=use_weights, weight_alpha=args.weight_alpha
        )

        # Validate
        val_metrics = validate(
            model, val_loader, loss_fn, device,
            use_weights=use_weights, weight_alpha=args.weight_alpha
        )

        # Update learning rate
        scheduler.step()

        # Record history
        history['train_loss'].append(train_metrics['train_loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['PRD'].append(val_metrics['PRD'])
        history['WWPRD'].append(val_metrics['WWPRD'])
        if 'PRD_vs_noisy' in val_metrics:
            history['PRD_vs_noisy'].append(val_metrics['PRD_vs_noisy'])
        if 'WWPRD_vs_noisy' in val_metrics:
            history['WWPRD_vs_noisy'].append(val_metrics['WWPRD_vs_noisy'])
        history['SNR_out'].append(val_metrics['SNR_out'])
        if 'SNR_in' in val_metrics:
            history['SNR_in'].append(val_metrics['SNR_in'])
            history['SNR_improvement'].append(val_metrics['SNR_improvement'])

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            console.print(f"\n[bold]Epoch {epoch}/{args.epochs}")
            console.print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            console.print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            console.print(f"  PRD (vs clean): {val_metrics['PRD']:.2f}%")
            if 'PRD_vs_noisy' in val_metrics:
                console.print(f"  PRD (vs noisy): {val_metrics['PRD_vs_noisy']:.2f}%")
            console.print(f"  WWPRD (vs clean): {val_metrics['WWPRD']:.2f}%")
            if 'SNR_improvement' in val_metrics:
                console.print(f"  SNR Improv: {val_metrics['SNR_improvement']:.2f} dB")

        # Save best model
        if val_metrics['val_loss'] < best_val_loss and args.save_model:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
            }, output_dir / "best_model.pth")

    # Final evaluation
    console.print("\n[bold green]Final Evaluation:")
    final_metrics = validate(
        model, val_loader, loss_fn, device,
        use_weights=use_weights, weight_alpha=args.weight_alpha
    )
    console.print(format_metrics(final_metrics, "Final Metrics"))

    # Calculate compression ratio
    signal_length = 360 * args.window_seconds  # samples
    original_bits = 11  # bits per sample
    latent_bits = 8  # quantization bits
    cr = compute_compression_ratio(
        signal_length * original_bits,
        args.latent_dim * (signal_length // (2 ** len(args.hidden_dims))) * latent_bits
    )
    final_metrics['CR'] = cr

    # Save final metrics
    with open(output_dir / "final_metrics.json", "w") as f:
        serializable_metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                               for k, v in final_metrics.items()}
        json.dump(serializable_metrics, f, indent=2)

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_curves(history, output_dir)

    # Plot reconstruction examples
    plot_reconstruction_examples(model, val_loader, device, output_dir)

    console.print(f"\n[bold green]✓ Ablation training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

