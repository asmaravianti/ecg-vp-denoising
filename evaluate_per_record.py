"""Generate per-record evaluation for all MIT-BIH records"""
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import track

from ecgdae.data import MITBIHDataset, NSTDBNoiseMixer, WindowingConfig
from ecgdae.models import ResidualAutoEncoder, ConvAutoEncoder
from ecgdae.metrics import compute_prd, compute_wwprd_wavelet, compute_snr
from ecgdae.quantization import quantize_latent, dequantize_latent, compute_compression_ratio

console = Console()

def evaluate_per_record(model_path: str, config_path: str, quantization_bits: int = 4):
    """Evaluate model performance per record"""

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

    console.print(f"✓ Loaded model: latent_dim={config['latent_dim']}")

    # Load all MIT-BIH records
    data_dir = Path("./data/mitbih")
    noise_dir = Path("./data/nstdb")

    # Standard test records
    all_records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]

    # Initialize NSTDB noise mixer (same API as training/evaluation)
    nstdb = NSTDBNoiseMixer(data_dir=str(noise_dir))

    results_per_record = []

    console.print(f"\n[bold]Evaluating {len(all_records)} MIT-BIH records...")

    for record_id in track(all_records, description="Processing records"):
        try:
            # Create per-record noise mixer
            snr_db = config.get("snr_db", 10.0)
            noise_mixer = nstdb.create_mixer(
                target_snr_db=snr_db,
                noise_type=config.get("nstdb_noise", "muscle_artifact"),
            )

            # Load dataset for this record
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
                console.print(f"[yellow]⚠ Record {record_id} not found or empty")
                continue

            # Create dataloader
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

            # Evaluate
            prd_list = []
            wwprd_list = []
            snr_in_list = []
            snr_out_list = []

            with torch.no_grad():
                for batch_idx, (noisy, clean) in enumerate(dataloader):
                    noisy = noisy.to(device)
                    clean = clean.to(device)

                    # Forward pass
                    latent = model.encode(noisy)

                    # Quantize
                    min_val = latent.min().item()
                    max_val = latent.max().item()
                    quantized, scale, offset = quantize_latent(
                        latent, quantization_bits, min_val, max_val
                    )
                    dequantized = dequantize_latent(
                        quantized, scale, offset, quantization_bits
                    )

                    # Decode
                    recon = model.decode(dequantized)

                    # Compute metrics
                    for i in range(clean.size(0)):
                        clean_sig = clean[i, 0].cpu().numpy()
                        noisy_sig = noisy[i, 0].cpu().numpy()
                        recon_sig = recon[i, 0].cpu().numpy()

                        prd = compute_prd(clean_sig, recon_sig)
                        wwprd = compute_wwprd_wavelet(clean_sig, recon_sig, alpha=2.0)
                        snr_in = compute_snr(clean_sig, noisy_sig)
                        snr_out = compute_snr(clean_sig, recon_sig)

                        prd_list.append(prd)
                        wwprd_list.append(wwprd)
                        snr_in_list.append(snr_in)
                        snr_out_list.append(snr_out)

            # Compute compression ratio
            window_samples = int(config.get('window_seconds', 2.0) * config.get('sample_rate', 360))
            cr = compute_compression_ratio(
                config['latent_dim'],
                window_samples,
                quantization_bits
            )

            # Aggregate results
            prd_mean = np.mean(prd_list)
            wwprd_mean = np.mean(wwprd_list)
            snr_improvement = np.mean(snr_out_list) - np.mean(snr_in_list)
            qs = cr / (prd_mean / 100.0)
            qsn = cr / (prd_mean / 100.0) * (1 + snr_improvement / 10.0)  # Normalized QS

            results_per_record.append({
                'Record': record_id,
                'PRD (%)': prd_mean,
                'WWPRD (%)': wwprd_mean,
                'SNR_improvement (dB)': snr_improvement,
                'CR': cr,
                'QS': qs,
                'QSN': qsn,
                'Num_Windows': len(prd_list)
            })

        except Exception as e:
            console.print(f"[red]✗ Error processing record {record_id}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results_per_record)

    # Display table
    console.print("\n" + "=" * 100)
    console.print("[bold]Per-Record Evaluation Results")
    console.print("=" * 100)
    console.print(df.to_string(index=False))

    # Summary statistics
    console.print("\n" + "=" * 100)
    console.print("[bold]Summary Statistics")
    console.print("=" * 100)
    console.print(f"Mean PRD: {df['PRD (%)'].mean():.2f}% ± {df['PRD (%)'].std():.2f}%")
    console.print(f"Mean WWPRD: {df['WWPRD (%)'].mean():.2f}% ± {df['WWPRD (%)'].std():.2f}%")
    console.print(f"Mean QS: {df['QS'].mean():.4f} ± {df['QS'].std():.4f}")
    console.print(f"CR: {df['CR'].iloc[0]:.2f}:1")

    # Highlight specific records
    console.print("\n" + "=" * 100)
    console.print("[bold]Records 117 and 119 (as requested)")
    console.print("=" * 100)
    for record_id in ['117', '119']:
        record_data = df[df['Record'] == record_id]
        if not record_data.empty:
            console.print(f"\nRecord {record_id}:")
            console.print(f"  PRD: {record_data['PRD (%)'].values[0]:.2f}%")
            console.print(f"  WWPRD: {record_data['WWPRD (%)'].values[0]:.2f}%")
            console.print(f"  CR: {record_data['CR'].values[0]:.2f}:1")
            console.print(f"  QS: {record_data['QS'].values[0]:.4f}")

    # Save results
    output_dir = Path("outputs/per_record_evaluation")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / f"latent{config['latent_dim']}_per_record.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"\n✓ Results saved to: {csv_path}")

    json_path = output_dir / f"latent{config['latent_dim']}_per_record.json"
    with open(json_path, 'w') as f:
        json.dump(results_per_record, f, indent=2)
    console.print(f"✓ Results saved to: {json_path}")

    return df

if __name__ == "__main__":
    # Evaluate best model (latent_dim=2)
    model_path = "outputs/wwprd_latent2_qat_optimized/best_model.pth"
    config_path = "outputs/wwprd_latent2_qat_optimized/config.json"

    df = evaluate_per_record(model_path, config_path, quantization_bits=4)

