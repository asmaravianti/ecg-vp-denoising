"""Compare WWPRD-only loss vs Combined loss (PRDN + WWPRD).

This script:
1. Trains model with WWPRD-only loss
2. Trains model with Combined loss (alpha*PRDN + (1-alpha)*WWPRD)
3. Evaluates both models on records 117 and 119
4. Generates comparison report

This addresses Professor's requirement:
"It would be nice to see whether the simple WWPRDN loss or the combined one works better"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def run_training(
    loss_type: str,
    combined_alpha: float = None,
    epochs: int = 50,
    base_config: dict = None,
    output_suffix: str = "",
) -> Path:
    """Run training with specified loss function.

    Returns:
        Path to output directory
    """
    output_dir = Path(f"./outputs/loss_comparison_{loss_type}{output_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "scripts/train_mitbih.py",
        "--data_dir", "./data/mitbih",
        "--num_records", str(base_config.get("num_records", 20)),
        "--window_seconds", str(base_config.get("window_seconds", 2.0)),
        "--sample_rate", str(base_config.get("sample_rate", 360)),
        "--noise_type", base_config.get("noise_type", "nstdb"),
        "--snr_db", str(base_config.get("snr_db", 10.0)),
        "--nstdb_noise", base_config.get("nstdb_noise", "muscle_artifact"),
        "--model_type", base_config.get("model_type", "residual"),
        "--hidden_dims"] + [str(d) for d in base_config.get("hidden_dims", [32, 64, 128])] + [
        "--latent_dim", str(base_config.get("latent_dim", 32)),
        "--loss_type", loss_type,
        "--weight_alpha", str(base_config.get("weight_alpha", 2.0)),
        "--batch_size", str(base_config.get("batch_size", 32)),
        "--epochs", str(epochs),
        "--lr", str(base_config.get("lr", 0.0005)),
        "--weight_decay", str(base_config.get("weight_decay", 0.0001)),
        "--val_split", str(base_config.get("val_split", 0.15)),
        "--output_dir", str(output_dir),
        "--save_model",
        "--device", "auto",
    ]

    if loss_type == "combined" and combined_alpha is not None:
        cmd.extend(["--combined_alpha", str(combined_alpha)])

    console.print(f"\n[bold cyan]{'='*60}")
    console.print(f"[bold cyan]Training with {loss_type} loss")
    if combined_alpha is not None:
        console.print(f"[cyan]Combined alpha: {combined_alpha}")
    console.print(f"[cyan]Output: {output_dir}")
    console.print(f"[bold cyan]{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[red]Training failed!")
        console.print(f"[red]Error: {result.stderr}")
        return None

    console.print(f"[green]✓ Training completed: {output_dir}")
    return output_dir


def evaluate_model(model_path: Path, config_path: Path) -> dict:
    """Evaluate model on records 117 and 119."""
    cmd = [
        "python", "scripts/evaluate_records_117_119.py",
        "--model_path", str(model_path),
        "--config_path", str(config_path),
        "--data_dir", "./data/mitbih",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[red]Evaluation failed: {result.stderr}")
        return None

    # Load evaluation results
    eval_path = model_path.parent / "evaluation_records_117_119.json"
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            return json.load(f)
    return None


def load_training_results(output_dir: Path) -> dict:
    """Load training results from output directory."""
    results = {}

    # Load final metrics
    final_metrics_path = output_dir / "final_metrics.json"
    if final_metrics_path.exists():
        with open(final_metrics_path, 'r') as f:
            results['final_metrics'] = json.load(f)

    # Load training history
    history_path = output_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            results['history'] = json.load(f)

    # Load config
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)

    return results


def print_comparison_report(
    wwprd_results: dict,
    combined_results: dict,
    wwprd_eval: dict = None,
    combined_eval: dict = None,
):
    """Print comprehensive comparison report."""

    # Training metrics comparison
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Training Metrics Comparison",
        border_style="cyan"
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("WWPRD Only", justify="right", style="green")
    table.add_column("Combined Loss", justify="right", style="yellow")
    table.add_column("Difference", justify="right", style="blue")

    wwprd_metrics = wwprd_results.get('final_metrics', {})
    combined_metrics = combined_results.get('final_metrics', {})

    # PRDN
    wwprd_prdn = wwprd_metrics.get('PRDN', 0)
    combined_prdn = combined_metrics.get('PRDN', 0)
    diff_prdn = combined_prdn - wwprd_prdn
    table.add_row(
        "PRDN (%)",
        f"{wwprd_prdn:.2f}",
        f"{combined_prdn:.2f}",
        f"{diff_prdn:+.2f}",
    )

    # WWPRD
    wwprd_wwprd = wwprd_metrics.get('WWPRD', 0)
    combined_wwprd = combined_metrics.get('WWPRD', 0)
    diff_wwprd = combined_wwprd - wwprd_wwprd
    table.add_row(
        "WWPRD (%)",
        f"{wwprd_wwprd:.2f}",
        f"{combined_wwprd:.2f}",
        f"{diff_wwprd:+.2f}",
    )

    # Validation Loss
    wwprd_val_loss = wwprd_metrics.get('val_loss', 0)
    combined_val_loss = combined_metrics.get('val_loss', 0)
    diff_val_loss = combined_val_loss - wwprd_val_loss
    table.add_row(
        "Val Loss",
        f"{wwprd_val_loss:.4f}",
        f"{combined_val_loss:.4f}",
        f"{diff_val_loss:+.4f}",
    )

    console.print(table)

    # Records 117 and 119 evaluation
    if wwprd_eval and combined_eval:
        console.print("\n")
        console.print(Panel.fit(
            "[bold cyan]Evaluation on Records 117 & 119",
            border_style="cyan"
        ))

        table2 = Table(show_header=True, header_style="bold magenta")
        table2.add_column("Record", style="cyan")
        table2.add_column("Metric", style="cyan")
        table2.add_column("WWPRD Only", justify="right", style="green")
        table2.add_column("Combined Loss", justify="right", style="yellow")
        table2.add_column("Better", justify="center", style="bold")

        for record in ['117', '119']:
            wwprd_rec = wwprd_eval.get(record, {})
            combined_rec = combined_eval.get(record, {})

            # PRDN
            wwprd_prdn = wwprd_rec.get('PRDN_mean', 0)
            combined_prdn = combined_rec.get('PRDN_mean', 0)
            better = "Combined" if combined_prdn < wwprd_prdn else "WWPRD"
            table2.add_row(
                record,
                "PRDN (%)",
                f"{wwprd_prdn:.2f}",
                f"{combined_prdn:.2f}",
                f"[bold green]{better}[/bold green]" if better == "Combined" else f"[bold yellow]{better}[/bold yellow]",
            )

            # WWPRD
            wwprd_wwprd = wwprd_rec.get('WWPRD_mean', 0)
            combined_wwprd = combined_rec.get('WWPRD_mean', 0)
            if not (np.isnan(wwprd_wwprd) or np.isnan(combined_wwprd)):
                better = "Combined" if combined_wwprd < wwprd_wwprd else "WWPRD"
                table2.add_row(
                    record,
                    "WWPRD (%)",
                    f"{wwprd_wwprd:.2f}",
                    f"{combined_wwprd:.2f}",
                    f"[bold green]{better}[/bold green]" if better == "Combined" else f"[bold yellow]{better}[/bold yellow]",
                )

        console.print(table2)

    # Summary
    console.print("\n")
    console.print(Panel.fit(
        "[bold yellow]Summary[/bold yellow]\n\n"
        "This comparison addresses Professor's requirement:\n"
        "'It would be nice to see whether the simple WWPRDN loss or the combined one works better'\n\n"
        "Results can be used in the experiment section of the TDK report.",
        border_style="yellow"
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Compare WWPRD-only vs Combined loss functions"
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for each training")
    parser.add_argument("--combined_alpha", type=float, default=0.5,
                        help="Alpha value for combined loss (default: 0.5)")
    parser.add_argument("--base_config", type=str, default=None,
                        help="Path to base config JSON (uses default if not provided)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only evaluate existing models")
    parser.add_argument("--wwprd_model", type=str, default=None,
                        help="Path to existing WWPRD model (if skip_training)")
    parser.add_argument("--combined_model", type=str, default=None,
                        help="Path to existing Combined model (if skip_training)")

    args = parser.parse_args()

    # Load base config
    if args.base_config:
        with open(args.base_config, 'r') as f:
            base_config = json.load(f)
    else:
        # Default config
        base_config = {
            "num_records": 20,
            "window_seconds": 2.0,
            "sample_rate": 360,
            "noise_type": "nstdb",
            "snr_db": 10.0,
            "nstdb_noise": "muscle_artifact",
            "model_type": "residual",
            "hidden_dims": [32, 64, 128],
            "latent_dim": 32,
            "weight_alpha": 2.0,
            "batch_size": 32,
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "val_split": 0.15,
        }

    console.print("[bold green]Loss Function Comparison Experiment")
    console.print(f"[cyan]Epochs per training: {args.epochs}")
    console.print(f"[cyan]Combined loss alpha: {args.combined_alpha}")

    wwprd_output = None
    combined_output = None

    if not args.skip_training:
        # Train WWPRD-only model
        console.print("\n[bold yellow]Step 1: Training with WWPRD-only loss")
        wwprd_output = run_training(
            loss_type="wwprd",
            epochs=args.epochs,
            base_config=base_config,
            output_suffix="",
        )

        if wwprd_output is None:
            console.print("[red]WWPRD training failed. Exiting.")
            return

        # Train Combined loss model
        console.print("\n[bold yellow]Step 2: Training with Combined loss")
        combined_output = run_training(
            loss_type="combined",
            combined_alpha=args.combined_alpha,
            epochs=args.epochs,
            base_config=base_config,
            output_suffix=f"_alpha{args.combined_alpha}",
        )

        if combined_output is None:
            console.print("[red]Combined loss training failed. Exiting.")
            return
    else:
        # Use existing models
        if args.wwprd_model:
            wwprd_output = Path(args.wwprd_model).parent
        if args.combined_model:
            combined_output = Path(args.combined_model).parent

    # Evaluate both models
    console.print("\n[bold yellow]Step 3: Evaluating on records 117 and 119")

    wwprd_eval = None
    combined_eval = None

    if wwprd_output:
        console.print(f"[cyan]Evaluating WWPRD model: {wwprd_output}")
        wwprd_eval = evaluate_model(
            wwprd_output / "best_model.pth",
            wwprd_output / "config.json",
        )

    if combined_output:
        console.print(f"[cyan]Evaluating Combined model: {combined_output}")
        combined_eval = evaluate_model(
            combined_output / "best_model.pth",
            combined_output / "config.json",
        )

    # Load training results
    wwprd_results = load_training_results(wwprd_output) if wwprd_output else {}
    combined_results = load_training_results(combined_output) if combined_output else {}

    # Print comparison report
    print_comparison_report(
        wwprd_results,
        combined_results,
        wwprd_eval,
        combined_eval,
    )

    # Save summary
    summary_path = Path("./outputs/loss_comparison_summary.json")
    summary = {
        "wwprd_training": wwprd_results,
        "combined_training": combined_results,
        "wwprd_evaluation": wwprd_eval,
        "combined_evaluation": combined_eval,
        "config": base_config,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]✓ Summary saved to {summary_path}")
    console.print("\n[bold cyan]Experiment complete! Results ready for TDK report.")


if __name__ == "__main__":
    import numpy as np
    main()


