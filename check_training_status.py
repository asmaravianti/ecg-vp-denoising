"""Check training status for Person B"""
import torch
import json
from pathlib import Path

model_path = Path("outputs/wwprd_latent8_improved/best_model.pth")
config_path = Path("outputs/wwprd_latent8_improved/config.json")
metrics_path = Path("outputs/wwprd_latent8_improved/final_metrics.json")

print("=" * 60)
print("PERSON B - Training Status Check")
print("=" * 60)

if model_path.exists():
    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
    epoch = ckpt.get('epoch', 'N/A')
    val_loss = ckpt.get('val_loss', 'N/A')
    print(f"✓ Model exists")
    print(f"  Current epoch: {epoch}")
    print(f"  Validation loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"  Validation loss: {val_loss}")
else:
    print("✗ Model not found")

if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"\n✓ Config exists")
    print(f"  Target epochs: {config.get('epochs', 'N/A')}")
    print(f"  Current epoch: {epoch if model_path.exists() else 'N/A'}")

if metrics_path.exists():
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    print(f"\n✓ Final metrics available:")
    print(f"  PRD: {metrics.get('PRD', 0):.2f}%")
    print(f"  PRDN: {metrics.get('PRDN', 0):.2f}%")
    print(f"  WWPRD: {metrics.get('WWPRD', 0):.2f}%")
    print(f"  SNR improvement: {metrics.get('SNR_improvement', 0):.2f} dB")

print("\n" + "=" * 60)
