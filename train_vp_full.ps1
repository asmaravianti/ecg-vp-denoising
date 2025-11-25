# VP Layer Full Training Script
# Uses same configuration as baseline (48 records, 200 epochs) for fair comparison

Write-Host "========================================"
Write-Host "VP Layer Full Training (48 records, 200 epochs)"
Write-Host "Same config as baseline for fair comparison"
Write-Host "========================================"
Write-Host ""

# Training parameters (matching baseline best results)
$output_dir = "./outputs/vp_full_48records_200epochs"

Write-Host "Starting VP Model Training..."
Write-Host "Configuration:"
Write-Host "  - Records: 48 (full dataset)"
Write-Host "  - Epochs: 200"
Write-Host "  - Latent Dim: 2"
Write-Host "  - QAT: Enabled (probability=0.7)"
Write-Host "  - Loss: WWPRD (alpha=2.0)"
Write-Host "  - Output: $output_dir"
Write-Host ""

# Train VP Model
python scripts/train_mitbih.py `
    --num_records 48 `
    --epochs 200 `
    --batch_size 32 `
    --loss_type wwprd `
    --weight_alpha 2.0 `
    --latent_dim 2 `
    --quantization_aware `
    --qat_probability 0.7 `
    --model_type vp `
    --output_dir $output_dir `
    --save_model

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Training completed successfully!"
    Write-Host ""
    Write-Host "Now evaluating compression performance..."

    # Evaluate Compression
    python scripts/evaluate_compression.py `
        --model_path "$output_dir/best_model.pth" `
        --config_path "$output_dir/config.json" `
        --output_file "$output_dir/compression_results.json" `
        --quantization_bits 4

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================"
        Write-Host "VP Full Training Complete!"
        Write-Host "Results saved to: $output_dir"
        Write-Host "========================================"
    } else {
        Write-Host "Error during compression evaluation"
    }
} else {
    Write-Host "Error during training"
}



