# Quick Test Training Script for PowerShell
# Fast training for testing (10 minutes)

Write-Host "Starting Quick Test Training..." -ForegroundColor Green
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  - Records: 5"
Write-Host "  - Epochs: 20"
Write-Host "  - Output: ./outputs/quick_test"
Write-Host ""

python scripts/train_mitbih.py `
    --num_records 5 `
    --epochs 20 `
    --batch_size 32 `
    --loss_type wwprd `
    --output_dir ./outputs/quick_test

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Quick test completed successfully!" -ForegroundColor Green
    Write-Host "Results saved to: ./outputs/quick_test/" -ForegroundColor Yellow
} else {
    Write-Host "Training failed. Please check the error messages above." -ForegroundColor Red
}

