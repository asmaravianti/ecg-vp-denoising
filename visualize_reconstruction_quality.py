"""Visualize reconstruction quality for sanity check (especially latent_dim=2)"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from ecgdae.data import MITBIHDataset, NSTDBNoiseMixer, WindowingConfig
from ecgdae.models import ResidualAutoEncoder, ConvAutoEncoder
from ecgdae.quantization import quantize_latent, dequantize_latent

def visualize_reconstructions(model_path: str, config_path: str, record_ids: list, num_samples: int = 3):
    """Visualize reconstructions for specific records"""
    import json

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device('cpu')

    # Load model based on model_type
    model_type = config.get('model_type', 'conv')
    if model_type == 'conv':
        model = ConvAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(config['hidden_dims']),
            latent_dim=config['latent_dim'],
        )
    else:  # residual
        model = ResidualAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(config['hidden_dims']),
            latent_dim=config['latent_dim'],
            num_res_blocks=config.get('num_res_blocks', 2),
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print(f"✓ Loaded model: latent_dim={config['latent_dim']}")

    # Load data
    data_dir = Path("./data/mitbih")
    noise_dir = Path("./data/nstdb")
    # Initialize NSTDB noise mixer (same API as training/evaluation)
    nstdb = NSTDBNoiseMixer(data_dir=str(noise_dir))

    fig, axes = plt.subplots(len(record_ids) * num_samples, 1,
                             figsize=(15, 4 * len(record_ids) * num_samples))
    if len(record_ids) * num_samples == 1:
        axes = [axes]

    plot_idx = 0

    for record_id in record_ids:
        print(f"\nProcessing record {record_id}...")

        # Create per-record noise mixer
        snr_db = config.get("snr_db", 10.0)
        noise_mixer = nstdb.create_mixer(
            target_snr_db=snr_db,
            noise_type=config.get("nstdb_noise", "muscle_artifact"),
        )

        # Load dataset
        dataset = MITBIHDataset(
            records=[record_id],
            config=WindowingConfig(
                window_seconds=config.get("window_seconds", 2.0),
                sample_rate=config.get("sample_rate", 360),
            ),
            noise_mixer=noise_mixer,
            data_dir=str(data_dir),
        )

        if len(dataset) == 0:
            print(f"⚠ Record {record_id} not found")
            continue

        # Get samples
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        for idx in indices:
            noisy, clean = dataset[idx]

            # Add batch dimension
            noisy_batch = noisy.unsqueeze(0).to(device)
            clean_batch = clean.unsqueeze(0).to(device)

            # Forward pass with quantization
            with torch.no_grad():
                latent = model.encode(noisy_batch)

                # Quantize (4-bit)
                min_val = latent.min().item()
                max_val = latent.max().item()
                quantized, scale, offset = quantize_latent(latent, 4, min_val, max_val)
                dequantized = dequantize_latent(quantized, scale, offset, 4)

                # Decode
                recon = model.decode(dequantized)

            # Convert to numpy
            clean_sig = clean[0].cpu().numpy()
            noisy_sig = noisy[0].cpu().numpy()
            recon_sig = recon[0, 0].cpu().numpy()

            # Compute metrics
            from ecgdae.metrics import compute_prd, compute_wwprd_wavelet
            prd = compute_prd(clean_sig, recon_sig)
            wwprd = compute_wwprd_wavelet(clean_sig, recon_sig, alpha=2.0)

            # Plot
            ax = axes[plot_idx]
            time = np.arange(len(clean_sig)) / config.get('sample_rate', 360)

            ax.plot(time, clean_sig, 'g-', label='Clean', linewidth=1.5, alpha=0.8)
            ax.plot(time, noisy_sig, 'gray', label='Noisy', linewidth=1.0, alpha=0.5)
            ax.plot(time, recon_sig, 'r--', label='Reconstructed', linewidth=1.5, alpha=0.8)

            ax.set_title(f'Record {record_id} - Sample {idx} | '
                        f'PRD={prd:.2f}%, WWPRD={wwprd:.2f}% | '
                        f'latent_dim={config["latent_dim"]}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Amplitude (normalized)', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

            plot_idx += 1

    plt.tight_layout()

    # Save figure
    output_dir = Path("outputs/per_record_evaluation")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"reconstruction_quality_latent{config['latent_dim']}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    return output_path

if __name__ == "__main__":
    # Visualize for records 117 and 119 (as requested by professor)
    model_path = "outputs/wwprd_latent2_qat_optimized/best_model.pth"
    config_path = "outputs/wwprd_latent2_qat_optimized/config.json"

    print("=" * 80)
    print("Reconstruction Quality Visualization (Sanity Check)")
    print("=" * 80)
    print("\nFocusing on records 117 and 119 as requested by professor")
    print("This checks if latent_dim=2 is oversimplifying the ECG signals")
    print()

    visualize_reconstructions(
        model_path,
        config_path,
        record_ids=['117', '119', '100'],  # Include 100 as reference
        num_samples=3
    )

    print("\n" + "=" * 80)
    print("Sanity Check Complete")
    print("=" * 80)
    print("\nPlease review the visualization to check if:")
    print("1. QRS complexes are preserved")
    print("2. P-waves and T-waves are visible")
    print("3. Overall morphology is maintained")
    print("\nIf the reconstruction looks poor, consider:")
    print("- Increasing latent_dim (e.g., 4 or 8)")
    print("- Extending training epochs")
    print("- Adjusting QAT parameters")

