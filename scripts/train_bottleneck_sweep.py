"""Bottleneck sweep script for Week 3.

Trains multiple models with different latent_dim values to verify
monotonic rate-distortion behavior.
"""

import argparse
import json
import subprocess
from pathlib import Path
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Bottleneck sweep for rate-distortion analysis")

    parser.add_argument("--latent_dims", type=int, nargs="+",
                        default=[8, 12, 16, 20, 24, 32, 40, 48],
                        help="List of latent dimensions to test")
    parser.add_argument("--base_config", type=str,
                        default="outputs/residual_attempt1_complete/config.json",
                        help="Base configuration file")
    parser.add_argument("--output_dir", type=str, default="./outputs/week3/bottleneck_sweep",
                        help="Base output directory")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of epochs per model")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")

    args = parser.parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Load base config if exists
    base_config = {}
    if Path(args.base_config).exists():
        with open(args.base_config, 'r') as f:
            base_config = json.load(f)

    console.print(f"[bold cyan]Bottleneck Sweep")
    console.print(f"[cyan]Latent dimensions: {args.latent_dims}")
    console.print(f"[cyan]Epochs per model: {args.epochs}")
    console.print(f"[cyan]Output directory: {output_base}")

    # Create commands table
    table = Table(title="Training Commands", show_header=True)
    table.add_column("Latent Dim", style="cyan")
    table.add_column("Output Dir", style="green")
    table.add_column("Status", style="yellow")

    commands = []

    for latent_dim in args.latent_dims:
        output_dir = output_base / f"latent_{latent_dim}"

        cmd = [
            "python", "scripts/train_mitbih.py",
            "--model_type", base_config.get("model_type", "residual"),
            "--num_records", str(base_config.get("num_records", 20)),
            "--epochs", str(args.epochs),
            "--batch_size", str(base_config.get("batch_size", 32)),
            "--lr", str(base_config.get("lr", 0.0005)),
            "--weight_decay", str(base_config.get("weight_decay", 0.0001)),
            "--latent_dim", str(latent_dim),
            "--loss_type", base_config.get("loss_type", "wwprd"),
            "--save_model",
            "--output_dir", str(output_dir),
        ]

        commands.append((latent_dim, output_dir, cmd))
        table.add_row(str(latent_dim), str(output_dir), "Pending")

    console.print("\n")
    console.print(table)

    if args.dry_run:
        console.print("\n[yellow]Dry run mode - commands:")
        for latent_dim, output_dir, cmd in commands:
            console.print(f"\n[cyan]Latent dim {latent_dim}:")
            console.print(f"  {' '.join(cmd)}")
        return

    # Ask for confirmation
    console.print(f"\n[yellow]This will train {len(commands)} models sequentially.")
    console.print("[yellow]Estimated time: {:.1f}-{:.1f} hours".format(
        len(commands) * 10, len(commands) * 16
    ))

    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        console.print("[red]Cancelled.")
        return

    # Execute commands
    results = []
    for i, (latent_dim, output_dir, cmd) in enumerate(commands, 1):
        console.print(f"\n[bold green]Training model {i}/{len(commands)}: latent_dim={latent_dim}")
        console.print(f"[cyan]Output: {output_dir}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            results.append((latent_dim, output_dir, "Success"))
            console.print(f"[green]✓ Completed latent_dim={latent_dim}")
        except subprocess.CalledProcessError as e:
            results.append((latent_dim, output_dir, f"Failed: {e}"))
            console.print(f"[red]✗ Failed latent_dim={latent_dim}")
            # Continue with next model

    # Summary
    console.print("\n[bold green]Summary:")
    summary_table = Table(show_header=True)
    summary_table.add_column("Latent Dim", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Output Dir", style="yellow")

    for latent_dim, output_dir, status in results:
        summary_table.add_row(str(latent_dim), status, str(output_dir))

    console.print(summary_table)

    # Generate summary JSON
    summary = {
        "latent_dims": args.latent_dims,
        "epochs": args.epochs,
        "results": [
            {
                "latent_dim": ld,
                "output_dir": str(od),
                "status": st
            }
            for ld, od, st in results
        ]
    }

    with open(output_base / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]✓ Summary saved to {output_base / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()

