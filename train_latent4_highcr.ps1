# PowerShell script: Train latent_dim=4 model for higher CR (Teammate A)
# Usage: .\train_latent4_highcr.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training latent_dim=4 model (High CR Strategy)" -ForegroundColor Yellow
Write-Host "Teammate A Task" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 4 `
    --epochs 100 `
    --num_records 20 `
    --batch_size 32 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --output_dir "outputs/wwprd_latent4_highcr" `
    --save_model `
    --device auto

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Training completed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next: Evaluate the model" -ForegroundColor Cyan
    Write-Host "  python scripts/evaluate_compression.py --model_path outputs/wwprd_latent4_highcr/best_model.pth --config_path outputs/wwprd_latent4_highcr/config.json --compression_ratios 8 16 32 --quantization_bits 4 --output_file outputs/week2/wwprd_latent4_results.json"
} else {
    Write-Host "Training failed!" -ForegroundColor Red
}
