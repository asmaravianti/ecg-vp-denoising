# PowerShell script: Train multiple models with different latent_dim
# Usage: .\train_multiple_latent_dims.ps1

$dims = @(8, 16, 32)

foreach ($dim in $dims) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Training model with latent_dim=$dim" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan

    python scripts/train_mitbih.py `
        --loss_type wwprd `
        --latent_dim $dim `
        --epochs 50 `
        --num_records 20 `
        --batch_size 32 `
        --lr 0.0005 `
        --weight_decay 0.0001 `
        --output_dir "outputs/wwprd_latent$dim" `
        --save_model `
        --device auto

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed: latent_dim=$dim" -ForegroundColor Red
        break
    }

    Write-Host "Completed: latent_dim=$dim" -ForegroundColor Green
    Write-Host ""
}

Write-Host "All models training completed!" -ForegroundColor Green
