"""Complete the loss function comparison experiment.

This script:
1. Trains WWPRD-only model to 50 epochs
2. Trains Combined Loss model to 50 epochs
3. Evaluates both models on records 117 and 119
4. Generates comparison report
"""
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_command(cmd, description):
    """Run a command and handle errors."""
    console.print(f"\n[bold cyan]{'='*60}")
    console.print(f"[bold cyan]{description}")
    console.print(f"[bold cyan]{'='*60}")
    console.print(f"[yellow]Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        console.print(f"[red]✗ {description} failed!")
        return False

    console.print(f"[green]✓ {description} completed!")
    return True

def main():
    console.print(Panel.fit(
        "[bold green]Loss Function Comparison Experiment[/bold green]\n\n"
        "This will:\n"
        "1. Train WWPRD-only model (50 epochs)\n"
        "2. Train Combined Loss model (50 epochs)\n"
        "3. Evaluate both models on records 117 & 119\n"
        "4. Generate comparison report\n\n"
        "[yellow]Estimated time: 7-9 hours on CPU[/yellow]",
        border_style="green"
    ))

    # Step 1: Train WWPRD-only model
    console.print("\n[bold yellow]Step 1/4: Training WWPRD-only model (50 epochs)")
    wwprd_cmd = [
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
        "--save_model",
        "--device", "auto",
    ]

    if not run_command(wwprd_cmd, "WWPRD-only training"):
        console.print("[red]Experiment failed at Step 1")
        return 1

    # Step 2: Train Combined Loss model
    console.print("\n[bold yellow]Step 2/4: Training Combined Loss model (50 epochs)")
    combined_cmd = [
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
        "--save_model",
        "--device", "auto",
    ]

    if not run_command(combined_cmd, "Combined Loss training"):
        console.print("[red]Experiment failed at Step 2")
        return 1

    # Step 3: Run comparison script
    console.print("\n[bold yellow]Step 3/4: Running comparison evaluation")
    comparison_cmd = [
        "python", "scripts/compare_loss_functions.py",
        "--epochs", "50",
        "--combined_alpha", "0.5",
        "--skip_training",  # Use existing models
        "--wwprd_model", "outputs/loss_comparison_wwprd/best_model.pth",
        "--combined_model", "outputs/loss_comparison_combined_alpha0.5/best_model.pth",
    ]

    if not run_command(comparison_cmd, "Comparison evaluation"):
        console.print("[red]Experiment failed at Step 3")
        return 1

    # Step 4: Summary
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]✓ Experiment Complete![/bold green]\n\n"
        "Results saved to:\n"
        "- outputs/loss_comparison_wwprd/\n"
        "- outputs/loss_comparison_combined_alpha0.5/\n"
        "- outputs/loss_comparison_summary.json\n\n"
        "You can now use these results in your TDK report!",
        border_style="green"
    ))

    return 0

if __name__ == "__main__":
    sys.exit(main())


