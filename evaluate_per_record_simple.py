"""Simple per-record evaluation for the actual test records used in compression.

This avoids Rich / unicode issues and only evaluates the records that were
actually used in `evaluate_compression.py` (5 test records after training set).
"""

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd

from ecgdae.data import (
    MITBIHDataset,
    NSTDBNoiseMixer,
    WindowingConfig,
    MITBIHLoader,
)
from ecgdae.models import ConvAutoEncoder, ResidualAutoEncoder
from ecgdae.metrics import compute_prd, compute_wwprd_wavelet, compute_snr
from ecgdae.quantization import (
    quantize_latent,
    dequantize_latent,
    compute_compression_ratio,
)


def load_model(model_path: str, config: Dict, device: torch.device):
    """Load Conv or Residual autoencoder from checkpoint."""
    model_type = config.get("model_type", "conv")
    if model_type == "conv":
        model = ConvAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(config["hidden_dims"]),
            latent_dim=config["latent_dim"],
        )
    else:
        model = ResidualAutoEncoder(
            in_channels=1,
            hidden_dims=tuple(config["hidden_dims"]),
            latent_dim=config["latent_dim"],
            num_res_blocks=config.get("num_res_blocks", 2),
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_all_records() -> List[str]:
    """Return the full list of 48 MIT-BIH records used in this project."""
    return MITBIHLoader.MITBIH_RECORDS


def evaluate_per_record_simple(
    model_path: str,
    config_path: str,
    quantization_bits: int = 4,
    num_windows_per_record: int = 200,
):
    """Evaluate PRD / WWPRD / QS per record for the actual test records."""
    # Load config and model
    with open(config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cpu")
    model = load_model(model_path, config, device)

    # Data / noise setup (same as evaluate_compression)
    window_config = WindowingConfig(
        window_seconds=config.get("window_seconds", 2.0),
        sample_rate=config.get("sample_rate", 360),
    )

    if config.get("noise_type", "nstdb") == "gaussian":
        from ecgdae.data import gaussian_snr_mixer

        noise_mixer = gaussian_snr_mixer(config.get("snr_db", 10.0))
    else:
        nstdb = NSTDBNoiseMixer(data_dir="./data/nstdb")
        noise_mixer = nstdb.create_mixer(
            target_snr_db=config.get("snr_db", 10.0),
            noise_type=config.get("nstdb_noise", "muscle_artifact"),
        )

    # Evaluate all 48 records (as requested by professor)
    all_records = get_all_records()

    results = []

    for record_id in all_records:
        # Build dataset for this record only
        dataset = MITBIHDataset(
            records=[record_id],
            config=window_config,
            noise_mixer=noise_mixer,
            data_dir=config.get("data_dir", "./data/mitbih"),
        )
        if len(dataset) == 0:
            print(f"[WARN] Record {record_id} dataset is empty, skipping.")
            continue

        # For speed, sample a subset of windows per record
        if num_windows_per_record is not None and len(dataset) > num_windows_per_record:
            indices = torch.randperm(len(dataset))[:num_windows_per_record]
            subset = Subset(dataset, indices.tolist())
        else:
            subset = dataset

        loader = DataLoader(subset, batch_size=32, shuffle=False)

        prd_vals = []
        wwprd_vals = []
        snr_in_vals = []
        snr_out_vals = []

        with torch.no_grad():
            for noisy, clean in loader:
                noisy = noisy.to(device)
                clean = clean.to(device)

                # Encode
                latent = model.encode(noisy)

                # Quantize latent (reuse helper, returns metadata)
                quantized, metadata = quantize_latent(
                    latent, quantization_bits, return_metadata=True
                )
                dequantized = dequantize_latent(quantized, metadata)

                # Decode
                recon = model.decode(dequantized)

                for i in range(clean.size(0)):
                    clean_sig = clean[i, 0].cpu().numpy()
                    noisy_sig = noisy[i, 0].cpu().numpy()
                    recon_sig = recon[i, 0].cpu().numpy()

                    prd_vals.append(compute_prd(clean_sig, recon_sig))
                    wwprd_vals.append(compute_wwprd_wavelet(clean_sig, recon_sig))
                    snr_in_vals.append(compute_snr(clean_sig, noisy_sig))
                    snr_out_vals.append(compute_snr(clean_sig, recon_sig))

        if not prd_vals:
            print(f"[WARN] No windows evaluated for record {record_id}, skipping.")
            continue

        prd_mean = float(np.mean(prd_vals))
        wwprd_mean = float(np.mean(wwprd_vals))
        snr_in = float(np.mean(snr_in_vals))
        snr_out = float(np.mean(snr_out_vals))
        snr_improvement = snr_out - snr_in

        # Compute CR and QS (use helper with shapes)
        window_samples = int(
            config.get("window_seconds", 2.0) * config.get("sample_rate", 360)
        )
        original_shape = (window_samples,)
        # Latent shape: (channels=latent_dim, length=window_samples / overall_downsample)
        # We approximate length from model's encoder stride: for our model,
        # effective downsample factor is 16 (2^4) so latent_length ≈ window_samples / 16.
        downsample = 16
        latent_length = max(1, window_samples // downsample)
        latent_shape = (config["latent_dim"], latent_length)
        cr = compute_compression_ratio(original_shape, latent_shape, quantization_bits)
        qs = cr / (prd_mean / 100.0)

        results.append(
            {
                "Record": record_id,
                "PRD (%)": prd_mean,
                "WWPRD (%)": wwprd_mean,
                "SNR_in (dB)": snr_in,
                "SNR_out (dB)": snr_out,
                "SNR_improvement (dB)": snr_improvement,
                "CR": cr,
                "QS": qs,
            }
        )

    if not results:
        print("No records were successfully evaluated.")
        return

    df = pd.DataFrame(results)
    print("\nPer-record results for all MIT-BIH records:")
    print(df.to_string(index=False))

    out_dir = Path("outputs/per_record_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "latent2_all_records.csv"
    json_path = out_dir / "latent2_all_records.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved JSON to: {json_path}")


if __name__ == "__main__":
    MODEL_PATH = "outputs/wwprd_latent2_qat_optimized/best_model.pth"
    CONFIG_PATH = "outputs/wwprd_latent2_qat_optimized/config.json"
    evaluate_per_record_simple(MODEL_PATH, CONFIG_PATH, quantization_bits=4)


