"""Check training status and results."""
import torch
import json
from pathlib import Path

# Check WWPRD model
wwprd_path = Path("outputs/loss_comparison_wwprd/best_model.pth")
if wwprd_path.exists():
    print("=" * 60)
    print("WWPRD Model Status:")
    print("=" * 60)
    ckpt = torch.load(str(wwprd_path), map_location='cpu', weights_only=False)
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Val Loss: {ckpt.get('val_loss', 'N/A'):.4f}")
    val_metrics = ckpt.get('val_metrics', {})
    print(f"Val Metrics:")
    for k, v in val_metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
else:
    print("WWPRD model not found")

# Check for training history
history_path = Path("outputs/loss_comparison_wwprd/training_history.json")
if history_path.exists():
    print("\nTraining History found!")
    with open(history_path) as f:
        history = json.load(f)
    print(f"Total epochs: {len(history.get('train_loss', []))}")
    if history.get('train_loss'):
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
else:
    print("\nNo training history found - training may not have completed")

# Check for combined model
combined_dirs = list(Path("outputs").glob("loss_comparison_combined*"))
if combined_dirs:
    print("\n" + "=" * 60)
    print("Combined Model Status:")
    print("=" * 60)
    for d in combined_dirs:
        model_path = d / "best_model.pth"
        if model_path.exists():
            ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
            print(f"\n{d.name}:")
            print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
            print(f"  Val Loss: {ckpt.get('val_loss', 'N/A'):.4f}")
else:
    print("\n" + "=" * 60)
    print("Combined Model: Not found")
    print("=" * 60)

# Check for summary
summary_path = Path("outputs/loss_comparison_summary.json")
if summary_path.exists():
    print("\n" + "=" * 60)
    print("Comparison Summary Found!")
    print("=" * 60)
    with open(summary_path) as f:
        summary = json.load(f)
    print("Summary keys:", list(summary.keys()))
else:
    print("\nNo comparison summary found")

