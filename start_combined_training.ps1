# PowerShell script to start Combined Loss training
# For teammate to run in parallel

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Combined Loss Model Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will train the Combined Loss model (50 epochs)" -ForegroundColor Yellow
Write-Host "Estimated time: 5-6 hours on CPU" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to cancel, or any key to start..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

python scripts/train_mitbih.py `
  --data_dir ./data/mitbih `
  --num_records 20 `
  --window_seconds 2.0 `
  --sample_rate 360 `
  --noise_type nstdb `
  --snr_db 10.0 `
  --nstdb_noise muscle_artifact `
  --model_type residual `
  --hidden_dims 32 64 128 `
  --latent_dim 32 `
  --loss_type combined `
  --combined_alpha 0.5 `
  --weight_alpha 2.0 `
  --batch_size 32 `
  --epochs 50 `
  --lr 0.0005 `
  --weight_decay 0.0001 `
  --val_split 0.15 `
  --output_dir outputs/loss_comparison_combined_alpha0.5 `
  --save_model `
  --device auto

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Training Completed Successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Results saved to: outputs/loss_comparison_combined_alpha0.5/" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Training Failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Yellow
}

