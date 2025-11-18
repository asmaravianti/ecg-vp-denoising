# PowerShell script: Extend latent_dim=8 training to reduce PRD (Teammate B)
# Usage: .\train_latent8_improved.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Extending latent_dim=8 training (Lower PRD Strategy)" -ForegroundColor Yellow
Write-Host "Teammate B Task" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "Resuming from: outputs/wwprd_latent8/best_model.pth" -ForegroundColor Yellow

python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 8 `
    --epochs 100 `
    --num_records 20 `
    --batch_size 32 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --output_dir "outputs/wwprd_latent8_improved" `
    --resume outputs/wwprd_latent8/best_model.pth `
    --save_model `
    --device auto

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Training completed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next: Evaluate the improved model" -ForegroundColor Cyan
    Write-Host "  python scripts/evaluate_compression.py --model_path outputs/wwprd_latent8_improved/best_model.pth --config_path outputs/wwprd_latent8_improved/config.json --compression_ratios 4 8 16 --quantization_bits 4 --output_file outputs/week2/wwprd_latent8_improved_results.json"
} else {
    Write-Host "Training failed!" -ForegroundColor Red
}
