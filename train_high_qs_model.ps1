# PowerShell script: Train model to achieve QS > 1
# Strategy: Use smaller latent_dim=4 and train longer (100 epochs)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training model for QS > 1" -ForegroundColor Yellow
Write-Host "Strategy: latent_dim=4, 100 epochs" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 4 `
    --epochs 100 `
    --num_records 20 `
    --batch_size 32 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --output_dir "outputs/wwprd_latent4_highqs" `
    --save_model `
    --device auto

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Training completed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: Evaluate with 4-bit quantization" -ForegroundColor Cyan
    Write-Host "  python scripts/evaluate_compression.py --model_path outputs/wwprd_latent4_highqs/best_model.pth --config_path outputs/wwprd_latent4_highqs/config.json --compression_ratios 4 8 16 32 --quantization_bits 4 --output_file outputs/week2/wwprd_latent4_results.json"
} else {
    Write-Host "Training failed!" -ForegroundColor Red
}



