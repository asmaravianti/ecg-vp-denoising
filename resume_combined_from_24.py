"""Resume Combined Loss training from Epoch 24 to 50."""
import subprocess
import sys
from pathlib import Path
from rich.console import Console
import torch

console = Console()

checkpoint = Path("outputs/loss_comparison_combined_alpha0.5/best_model.pth")
if not checkpoint.exists():
    console.print("[red]Checkpoint not found!")
    sys.exit(1)

# Check current epoch
ckpt = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
current_epoch = ckpt.get('epoch', 0)
console.print(f"[cyan]Current epoch: {current_epoch}/50")
console.print(f"[cyan]Best Val Loss: {ckpt.get('val_loss', 0):.4f}")

if current_epoch >= 50:
    console.print("[green]✓ Combined model training already complete!")
    sys.exit(0)

console.print(f"[bold green]Resuming Combined Loss training from Epoch {current_epoch} to 50...")
console.print(f"[cyan]Checkpoint: {checkpoint}")

cmd = [
    "python", "scripts/train_mitbih.py",
    "--data_dir", "./data/mitbih",
    "--num_records", "20",
    "--window_seconds", "2.0",
    "--sample_rate", "360",
    "--noise_type", "nstdb",
    "--snr_db", "10.0",
    "--nstdb_noise", "muscle_artifact",
    "--model_type", "residual",
    "--hidden_dims", "32", "64", "128",
    "--latent_dim", "32",
    "--loss_type", "combined",
    "--combined_alpha", "0.5",
    "--weight_alpha", "2.0",
    "--batch_size", "32",
    "--epochs", "50",
    "--lr", "0.0005",
    "--weight_decay", "0.0001",
    "--val_split", "0.15",
    "--output_dir", "outputs/loss_comparison_combined_alpha0.5",
    "--resume", str(checkpoint),
    "--save_model",
    "--device", "auto",
]

console.print("\n[bold yellow]Starting resumed training...")
console.print(f"[cyan]Will train from epoch {current_epoch + 1} to 50 ({50 - current_epoch} epochs remaining)")
result = subprocess.run(cmd)

if result.returncode == 0:
    console.print("\n[bold green]✓ Training completed!")
else:
    console.print("\n[bold red]✗ Training failed!")
    sys.exit(1)



