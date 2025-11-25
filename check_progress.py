"""Check current training progress."""
import json
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("CURRENT TRAINING PROGRESS")
print("=" * 60)
print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# WWPRD Model
wwprd_dir = Path("outputs/loss_comparison_wwprd")
if wwprd_dir.exists():
    print("📊 WWPRD-Only Model:")
    print("-" * 60)

    # Check checkpoint
    checkpoint = wwprd_dir / "best_model.pth"
    if checkpoint.exists():
        import torch
        ckpt = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', 0)
        val_loss = ckpt.get('val_loss', 0)
        val_metrics = ckpt.get('val_metrics', {})

        print(f"  Current Epoch: {epoch}/50 ({epoch*100//50}%)")
        print(f"  Best Val Loss: {val_loss:.4f}")
        print(f"  PRDN: {val_metrics.get('PRDN', 0):.2f}%")
        print(f"  WWPRD: {val_metrics.get('WWPRD', 0):.2f}%")
        print(f"  SNR Improvement: {val_metrics.get('SNR_improvement', 0):.2f} dB")
        print(f"  Last Updated: {datetime.fromtimestamp(checkpoint.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

    # Check history
    history_file = wwprd_dir / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        print(f"  History File: {len(history['train_loss'])} epochs recorded")
        if history['train_loss']:
            print(f"  Latest Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"  Latest Val Loss: {history['val_loss'][-1]:.4f}")
    print()

# Combined Model
combined_dir = Path("outputs/loss_comparison_combined_alpha0.5")
if combined_dir.exists():
    print("📊 Combined Loss Model:")
    print("-" * 60)

    checkpoint = combined_dir / "best_model.pth"
    if checkpoint.exists():
        import torch
        ckpt = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', 0)
        val_loss = ckpt.get('val_loss', 0)
        val_metrics = ckpt.get('val_metrics', {})

        print(f"  Current Epoch: {epoch}/50 ({epoch*100//50}%)")
        print(f"  Best Val Loss: {val_loss:.4f}")
        print(f"  PRDN: {val_metrics.get('PRDN', 0):.2f}%")
        print(f"  WWPRD: {val_metrics.get('WWPRD', 0):.2f}%")
        print(f"  Last Updated: {datetime.fromtimestamp(checkpoint.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

    history_file = combined_dir / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        print(f"  History File: {len(history['train_loss'])} epochs recorded")
else:
    print("📊 Combined Loss Model: Not started yet")
    print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
if wwprd_dir.exists() and (wwprd_dir / "best_model.pth").exists():
    import torch
    wwprd_epoch = torch.load(str(wwprd_dir / "best_model.pth"), map_location='cpu', weights_only=False).get('epoch', 0)
    print(f"WWPRD Model: {wwprd_epoch}/50 epochs ({wwprd_epoch*100//50}% complete)")

    if combined_dir.exists() and (combined_dir / "best_model.pth").exists():
        combined_epoch = torch.load(str(combined_dir / "best_model.pth"), map_location='cpu', weights_only=False).get('epoch', 0)
        print(f"Combined Model: {combined_epoch}/50 epochs ({combined_epoch*100//50}% complete)")
        remaining_wwprd = 50 - wwprd_epoch
        remaining_combined = 50 - combined_epoch
        print(f"\nRemaining: {remaining_wwprd} epochs (WWPRD) + {remaining_combined} epochs (Combined)")



