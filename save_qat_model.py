"""Quick script to save the QAT model from training history or re-save it."""

import json
import torch
from pathlib import Path
from ecgdae.models import ConvAutoEncoder
from ecgdae.data import MITBIHDataset, NSTDBNoiseMixer, WindowingConfig

def save_model_from_config():
    """Save model by re-initializing and loading from training history."""
    config_path = Path("outputs/wwprd_latent4_qat/config.json")
    output_dir = Path("outputs/wwprd_latent4_qat")

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load training history to get best epoch
    history_path = output_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)

        # Find best epoch (lowest val_loss)
        val_losses = history.get('val_loss', [])
        if val_losses:
            best_epoch = val_losses.index(min(val_losses)) + 1
            print(f"Best epoch from history: {best_epoch} (val_loss: {min(val_losses):.4f})")

    # Check for checkpoint files
    checkpoint_files = list(output_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoint_files:
        print(f"Found checkpoint files: {[f.name for f in checkpoint_files]}")
        # Use the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Using latest checkpoint: {latest_checkpoint.name}")

        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ConvAutoEncoder(
            in_channels=1,
            hidden_dims=config['hidden_dims'],
            latent_dim=config['latent_dim']
        ).to(device)

        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Save as best_model.pth
        torch.save({
            'epoch': checkpoint.get('epoch', best_epoch if 'best_epoch' in locals() else 150),
            'model_state_dict': model.state_dict(),
            'val_loss': checkpoint.get('val_loss', min(val_losses) if val_losses else None),
        }, output_dir / "best_model.pth")

        print(f"✅ Saved best_model.pth from {latest_checkpoint.name}")
        return True
    else:
        print("❌ No checkpoint files found. You need to re-run training with --save_model")
        return False

if __name__ == "__main__":
    save_model_from_config()

