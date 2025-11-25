"""Quick script to save VP model for evaluation"""
import torch
import json
from pathlib import Path
from ecgdae.models import VPAutoEncoder

# Load config
config_path = Path("outputs/vp_quick_test/config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# Create model
model = VPAutoEncoder(
    in_channels=1,
    hidden_dims=tuple(config['hidden_dims']),
    latent_dim=config['latent_dim'],
    num_res_blocks=2,
)

# Since we don't have saved weights, we'll need to retrain quickly
# But actually, let's just create a dummy checkpoint for now
# The real solution is to retrain with --save_model

print("Model created. Need to retrain with --save_model to get actual weights.")
print("Running quick retrain with save_model enabled...")




