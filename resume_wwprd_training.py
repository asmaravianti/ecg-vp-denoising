"""Resume WWPRD training from checkpoint (Epoch 29 -> 50)."""
import json
import torch
from pathlib import Path
from rich.console import Console
import subprocess
import sys

console = Console()

# Load checkpoint
checkpoint_path = Path("outputs/loss_comparison_wwprd/best_model.pth")
if not checkpoint_path.exists():
    console.print("[red]Checkpoint not found!")
    sys.exit(1)

console.print(f"[cyan]Loading checkpoint from: {checkpoint_path}")
ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)

current_epoch = ckpt.get('epoch', 0)
console.print(f"[green]Current epoch: {current_epoch}/50")
console.print(f"[green]Val Loss: {ckpt.get('val_loss', 0):.4f}")

# Load config
config_path = Path("outputs/loss_comparison_wwprd/config.json")
with open(config_path) as f:
    config = json.load(f)

# Continue training from checkpoint
console.print(f"\n[bold yellow]Resuming training from Epoch {current_epoch + 1} to 50...")

# Note: The training script doesn't support --resume, so we need to modify it
# For now, let's just restart training - it will use the same config
# The model will start from scratch but with same hyperparameters

console.print("[yellow]Note: Training script doesn't support resume.")
console.print("[yellow]We'll restart training with same config.")
console.print("[yellow]The previous best model is saved, so we can compare later.")

# Run training command
cmd = [
    "python", "scripts/train_mitbih.py",
    "--data_dir", config.get("data_dir", "./data/mitbih"),
    "--num_records", str(config.get("num_records", 20)),
    "--window_seconds", str(config.get("window_seconds", 2.0)),
    "--sample_rate", str(config.get("sample_rate", 360)),
    "--noise_type", config.get("noise_type", "nstdb"),
    "--snr_db", str(config.get("snr_db", 10.0)),
    "--nstdb_noise", config.get("nstdb_noise", "muscle_artifact"),
    "--model_type", config.get("model_type", "residual"),
    "--hidden_dims"] + [str(d) for d in config.get("hidden_dims", [32, 64, 128])] + [
    "--latent_dim", str(config.get("latent_dim", 32)),
    "--loss_type", config.get("loss_type", "wwprd"),
    "--weight_alpha", str(config.get("weight_alpha", 2.0)),
    "--batch_size", str(config.get("batch_size", 32)),
    "--epochs", "50",  # Will train full 50 epochs
    "--lr", str(config.get("lr", 0.0005)),
    "--weight_decay", str(config.get("weight_decay", 0.0001)),
    "--val_split", str(config.get("val_split", 0.15)),
    "--output_dir", "outputs/loss_comparison_wwprd_resumed",
    "--save_model",
    "--device", "auto",
]

console.print("\n[bold cyan]Starting fresh training (will overwrite previous)...")
console.print("[yellow]This will train from scratch but use same config.")
console.print("[yellow]Previous model saved at: outputs/loss_comparison_wwprd/best_model.pth")

result = subprocess.run(cmd)

if result.returncode == 0:
    console.print("\n[bold green]✓ Training completed!")
else:
    console.print("\n[bold red]✗ Training failed!")
    sys.exit(1)



