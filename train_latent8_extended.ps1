# PowerShell script: Continue training latent_dim=8 to 150 epochs (Person B)
# Usage: .\train_latent8_extended.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Extending latent_dim=8 training to 150 epochs" -ForegroundColor Yellow
Write-Host "Person B - Further PRD Reduction" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "Resuming from: outputs/wwprd_latent8_improved/best_model.pth" -ForegroundColor Yellow
Write-Host "Target: Reduce PRD from 29% to < 20%" -ForegroundColor Cyan
Write-Host ""

python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 8 `
    --epochs 150 `
    --num_records 20 `
    --batch_size 32 `
    --lr 0.0003 `
    --weight_decay 0.0001 `
    --output_dir "outputs/wwprd_latent8_150epochs" `
    --resume outputs/wwprd_latent8_improved/best_model.pth `
    --save_model `
    --device auto

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Training completed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "1. Evaluate the model:" -ForegroundColor Yellow
    Write-Host "   python scripts/evaluate_compression.py --model_path outputs/wwprd_latent8_150epochs/best_model.pth --config_path outputs/wwprd_latent8_150epochs/config.json --compression_ratios 4 8 16 --quantization_bits 4 --output_file outputs/week2/wwprd_latent8_150epochs_results.json" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Calculate QS scores:" -ForegroundColor Yellow
    Write-Host "   python calculate_qs_scores.py --results_dir outputs/week2 --output_dir outputs/week2" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "Training failed!" -ForegroundColor Red
}






