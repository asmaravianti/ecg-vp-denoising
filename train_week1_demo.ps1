# Week 1 Demo Training Script for PowerShell
# Standard training configuration (30-60 minutes)

Write-Host "Starting Week 1 Demo Training..." -ForegroundColor Green
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  - Records: 10"
Write-Host "  - Epochs: 50"
Write-Host "  - Loss: WWPRD"
Write-Host "  - Output: ./outputs/week1_demo"
Write-Host ""

python scripts/train_mitbih.py `
    --num_records 10 `
    --epochs 50 `
    --batch_size 32 `
    --loss_type wwprd `
    --weight_alpha 2.0 `
    --latent_dim 32 `
    --save_model `
    --output_dir ./outputs/week1_demo

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "Now running evaluation..." -ForegroundColor Cyan

    python scripts/evaluate_mitbih.py `
        --model_path ./outputs/week1_demo/best_model.pth `
        --config_path ./outputs/week1_demo/config.json `
        --output_dir ./outputs/week1_demo/evaluation `
        --num_samples 500

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Evaluation completed!" -ForegroundColor Green
        Write-Host "Results saved to: ./outputs/week1_demo/evaluation/" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Generated figures:" -ForegroundColor Cyan
        Write-Host "  - training_curves.png"
        Write-Host "  - reconstruction_examples.png"
        Write-Host "  - metric_distributions.png"
        Write-Host "  - quality_classification.png"
        Write-Host "  - prd_wwprd_scatter.png"
        Write-Host "  - reconstruction_gallery.png"
    }
} else {
    Write-Host "Training failed. Please check the error messages above." -ForegroundColor Red
}

