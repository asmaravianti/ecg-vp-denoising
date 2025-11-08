"""Training script for MIT-BIH ECG denoising and compression.

Week 1 deliverable:
- Train on MIT-BIH with NSTDB noise
- Optimize with differentiable WWPRD loss
- Generate training curves and evaluation metrics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from rich.console import Console
from rich.progress import track

from ecgdae.data import MITBIHDataset, NSTDBNoiseMixer, WindowingConfig, gaussian_snr_mixer
from ecgdae.losses import PRDLoss, WWPRDLoss, STFTWeightedWWPRDLoss
from ecgdae.models import ConvAutoEncoder, ResidualAutoEncoder, count_parameters
from ecgdae.metrics import batch_evaluate, format_metrics, compute_derivative_weights

console = Console()


def setup_args():
    parser = argparse.ArgumentParser(description="Train ECG autoencoder on MIT-BIH")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data/mitbih",
                        help="MIT-BIH data directory")
    parser.add_argument("--num_records", type=int, default=10,
                        help="Number of MIT-BIH records to use (for quick training)")
    parser.add_argument("--window_seconds", type=float, default=2.0,
                        help="Window length in seconds")
    parser.add_argument("--sample_rate", type=int, default=360,
                        help="Sampling rate in Hz")

    # Noise parameters
    parser.add_argument("--noise_type", type=str, default="nstdb",
                        choices=["gaussian", "nstdb"],
                        help="Type of noise to add")
    parser.add_argument("--snr_db", type=float, default=10.0,
                        help="Signal-to-noise ratio in dB")
    parser.add_argument("--nstdb_noise", type=str, default="muscle_artifact",
                        choices=["baseline_wander", "muscle_artifact", "electrode_motion"],
                        help="Type of NSTDB noise")

    # Model parameters
    parser.add_argument("--model_type", type=str, default="conv",
                        choices=["conv", "residual"],
                        help="Model architecture")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 64, 128],
                        help="Hidden dimensions for encoder")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Latent dimension (controls compression)")

    # Loss parameters
    parser.add_argument("--loss_type", type=str, default="wwprd",
                        choices=["mse", "prd", "wwprd", "stft_wwprd"],
                        help="Loss function type")
    parser.add_argument("--weight_alpha", type=float, default=2.0,
                        help="Alpha parameter for derivative-based WWPRD weights")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split ratio")

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs/week1",
                        help="Output directory for results")
    parser.add_argument("--save_model", action="store_true",
                        help="Save trained model")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use")

    return parser.parse_args()


def create_model(args) -> nn.Module:
    """Create model based on arguments."""
    if args.model_type == "conv":
        model = ConvAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(args.hidden_dims),
            latent_dim=args.latent_dim,
        )
    else:  # residual
        model = ResidualAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(args.hidden_dims),
            latent_dim=args.latent_dim,
            num_res_blocks=2,
        )

    console.print(f"[green]Model: {args.model_type}")
    console.print(f"[green]Parameters: {count_parameters(model):,}")

    return model


def create_loss_function(args) -> nn.Module:
    """Create loss function based on arguments."""
    if args.loss_type == "mse":
        return nn.MSELoss()
    elif args.loss_type == "prd":
        return PRDLoss()
    elif args.loss_type == "wwprd":
        return WWPRDLoss()
    elif args.loss_type == "stft_wwprd":
        return STFTWeightedWWPRDLoss()
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")


def create_dataloader(args) -> tuple:
    """Create train and validation dataloaders."""
    console.print("[yellow]Loading MIT-BIH dataset...")

    # Setup windowing config
    config = WindowingConfig(
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        step_seconds=args.window_seconds,  # Non-overlapping windows
    )

    # Setup noise mixer
    if args.noise_type == "gaussian":
        noise_mixer = gaussian_snr_mixer(args.snr_db)
    else:  # nstdb
        nstdb = NSTDBNoiseMixer(data_dir="./data/nstdb")
        noise_mixer = nstdb.create_mixer(
            target_snr_db=args.snr_db,
            noise_type=args.nstdb_noise,
            mix_gaussian=False,
        )

    # Use subset of records for faster training
    from ecgdae.data import MITBIHLoader
    all_records = MITBIHLoader.MITBIH_RECORDS
    selected_records = all_records[:args.num_records]

    console.print(f"[yellow]Using {len(selected_records)} records: {selected_records}")

    # Create dataset
    dataset = MITBIHDataset(
        records=selected_records,
        config=config,
        noise_mixer=noise_mixer,
        data_dir=args.data_dir,
        channel=0,
        normalize=True,
    )

    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    console.print(f"[green]Train samples: {len(train_dataset)}")
    console.print(f"[green]Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader


def compute_wwprd_weights(clean_batch: torch.Tensor, alpha: float) -> torch.Tensor:
    """Compute WWPRD weights for a batch of signals.

    Args:
        clean_batch: Clean signals (B, C, T)
        alpha: Scaling factor for derivative weights

    Returns:
        Weight tensor (B, C, T)
    """
    batch_size = clean_batch.shape[0]
    weights = []

    for i in range(batch_size):
        signal = clean_batch[i].cpu().numpy()
        w = compute_derivative_weights(signal, alpha=alpha)
        weights.append(torch.from_numpy(w))

    return torch.stack(weights).to(clean_batch.device)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_weights: bool = False,
    weight_alpha: float = 2.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for noisy, clean in train_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Forward pass
        recon = model(noisy)

        # Compute loss
        if use_weights and isinstance(loss_fn, WWPRDLoss):
            weights = compute_wwprd_weights(clean, weight_alpha)
            loss = loss_fn(clean, recon, weights)
        else:
            loss = loss_fn(clean, recon)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {"train_loss": total_loss / num_batches}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_weights: bool = False,
    weight_alpha: float = 2.0,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_clean = []
    all_recon = []
    all_noisy = []

    for noisy, clean in val_loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Forward pass
        recon = model(noisy)

        # Compute loss
        if use_weights and isinstance(loss_fn, WWPRDLoss):
            weights = compute_wwprd_weights(clean, weight_alpha)
            loss = loss_fn(clean, recon, weights)
        else:
            loss = loss_fn(clean, recon)

        total_loss += loss.item()
        num_batches += 1

        # Collect samples for metrics
        if len(all_clean) < 8:  # Collect 8 batches for metrics
            all_clean.append(clean)
            all_recon.append(recon)
            all_noisy.append(noisy)

    # Concatenate samples
    all_clean = torch.cat(all_clean, dim=0)
    all_recon = torch.cat(all_recon, dim=0)
    all_noisy = torch.cat(all_noisy, dim=0)

    # Compute detailed metrics
    metrics = batch_evaluate(all_clean, all_recon, all_noisy)
    metrics["val_loss"] = total_loss / num_batches

    return metrics


def plot_training_curves(history: Dict[str, List[float]], output_dir: Path):
    """Plot training loss curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training and validation loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # PRD over epochs
    ax = axes[0, 1]
    ax.plot(epochs, history['PRD'], 'g-', linewidth=2, label='PRD (vs clean)')
    if 'PRD_vs_noisy' in history and len(history['PRD_vs_noisy']) > 0:
        ax.plot(epochs, history['PRD_vs_noisy'], 'g--', linewidth=2, label='PRD (vs noisy)')
    ax.axhline(y=4.33, color='r', linestyle='--', label='Excellent threshold', alpha=0.7)
    ax.axhline(y=9.00, color='orange', linestyle='--', label='Very Good threshold', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('PRD (%)', fontsize=12)
    ax.set_title('PRD over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # WWPRD over epochs
    ax = axes[1, 0]
    ax.plot(epochs, history['WWPRD'], 'm-', linewidth=2, label='WWPRD (vs clean)')
    if 'WWPRD_vs_noisy' in history and len(history['WWPRD_vs_noisy']) > 0:
        ax.plot(epochs, history['WWPRD_vs_noisy'], 'm--', linewidth=2, label='WWPRD (vs noisy)')
    ax.axhline(y=7.4, color='r', linestyle='--', label='Excellent threshold', alpha=0.7)
    ax.axhline(y=14.8, color='orange', linestyle='--', label='Very Good threshold', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('WWPRD (%)', fontsize=12)
    ax.set_title('WWPRD over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # SNR improvement
    ax = axes[1, 1]
    if 'SNR_improvement' in history:
        ax.plot(epochs, history['SNR_improvement'], 'c-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('SNR Improvement (dB)', fontsize=12)
        ax.set_title('SNR Improvement over Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved training curves to {output_dir / 'training_curves.png'}")
    plt.close()


def plot_reconstruction_examples(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_examples: int = 4,
):
    """Plot reconstruction examples."""
    model.eval()

    # Get one batch
    noisy, clean = next(iter(val_loader))
    noisy = noisy.to(device)
    clean = clean.to(device)

    with torch.no_grad():
        recon = model(noisy)

    # Move to CPU for plotting
    noisy = noisy.cpu().numpy()
    clean = clean.cpu().numpy()
    recon = recon.cpu().numpy()

    # Plot examples
    fig, axes = plt.subplots(num_examples, 1, figsize=(14, 3*num_examples))

    for i in range(min(num_examples, len(clean))):
        ax = axes[i] if num_examples > 1 else axes

        time = np.arange(clean.shape[-1]) / 360.0  # Convert to seconds

        ax.plot(time, clean[i, 0], 'g-', label='Clean', linewidth=1.5, alpha=0.8)
        ax.plot(time, noisy[i, 0], 'gray', label='Noisy', linewidth=1, alpha=0.6)
        ax.plot(time, recon[i, 0], 'r--', label='Reconstructed', linewidth=1.5, alpha=0.8)

        # Compute PRD for this sample
        from ecgdae.metrics import compute_prd, compute_wwprd
        prd = compute_prd(clean[i], recon[i])
        wwprd = compute_wwprd(clean[i], recon[i])

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(f'Example {i+1} - PRD: {prd:.2f}%, WWPRD: {wwprd:.2f}%',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_examples.png", dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved reconstruction examples to {output_dir / 'reconstruction_examples.png'}")
    plt.close()


def main():
    args = setup_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    console.print(f"[bold cyan]Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create dataloaders
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
    console.print("\n[bold green]Starting training...")

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

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            console.print(f"\n[bold]Epoch {epoch}/{args.epochs}")
            console.print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            console.print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            console.print(f"  PRD (vs clean): {val_metrics['PRD']:.2f}% ({val_metrics.get('PRD_std', 0):.2f})")
            if 'PRD_vs_noisy' in val_metrics:
                console.print(f"  PRD (vs noisy): {val_metrics['PRD_vs_noisy']:.2f}% ({val_metrics.get('PRD_vs_noisy_std', 0):.2f})")
            console.print(f"  WWPRD (vs clean): {val_metrics['WWPRD']:.2f}% ({val_metrics.get('WWPRD_std', 0):.2f})")
            if 'WWPRD_vs_noisy' in val_metrics:
                console.print(f"  WWPRD (vs noisy): {val_metrics['WWPRD_vs_noisy']:.2f}% ({val_metrics.get('WWPRD_vs_noisy_std', 0):.2f})")
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

    # Save final metrics
    with open(output_dir / "final_metrics.json", "w") as f:
        # Convert to serializable format
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

    console.print(f"\n[bold green]✓ Training complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()

