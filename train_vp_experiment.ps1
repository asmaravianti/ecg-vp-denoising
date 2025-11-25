# VP Layer Experiment Script (Fixed)
# Compares VP Autoencoder vs Standard Residual Autoencoder (using best settings found so far)

# 1. Train VP Model (latent=2, QAT)
# Note: Using full command arguments to avoid PowerShell variable expansion issues
Write-Host "Training VP Model (latent=2, QAT)..."
python scripts/train_mitbih.py --num_records 48 --epochs 50 --batch_size 32 --loss_type wwprd --weight_alpha 2.0 --latent_dim 2 --quantization_aware --qat_probability 0.7 --model_type vp --output_dir ./outputs/vp_experiment_latent2

# 2. Evaluate Compression for VP Model
Write-Host "Evaluating VP Model Compression..."
python scripts/evaluate_compression.py --model_path ./outputs/vp_experiment_latent2/best_model.pth --config_path ./outputs/vp_experiment_latent2/config.json --output_file ./outputs/vp_experiment_latent2/compression_results.json --quantization_bits 4

Write-Host "VP Experiment Complete!"
