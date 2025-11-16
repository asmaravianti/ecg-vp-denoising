"""Resume WWPRD training from Epoch 29 to 50."""
import subprocess
import sys
from pathlib import Path
from rich.console import Console

console = Console()

checkpoint = Path("outputs/loss_comparison_wwprd/best_model.pth")
if not checkpoint.exists():
    console.print("[red]Checkpoint not found!")
    sys.exit(1)

console.print(f"[bold green]Resuming WWPRD training from Epoch 29...")
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
    "--loss_type", "wwprd",
    "--weight_alpha", "2.0",
    "--batch_size", "32",
    "--epochs", "50",
    "--lr", "0.0005",
    "--weight_decay", "0.0001",
    "--val_split", "0.15",
    "--output_dir", "outputs/loss_comparison_wwprd",
    "--resume", str(checkpoint),
    "--save_model",
    "--device", "auto",
]

console.print("\n[bold yellow]Starting resumed training...")
result = subprocess.run(cmd)

if result.returncode == 0:
    console.print("\n[bold green]✓ Training completed!")
else:
    console.print("\n[bold red]✗ Training failed!")
    sys.exit(1)



