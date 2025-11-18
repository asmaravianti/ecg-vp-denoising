# PowerShell script: Evaluate all models for compression ratio
# Usage: .\evaluate_all_models.ps1

$dims = @(8, 16, 32)
$quantization_bits = 4  # Use 4-bit quantization for higher CR

foreach ($dim in $dims) {
    $model_path = "outputs/wwprd_latent$dim/best_model.pth"
    $config_path = "outputs/wwprd_latent$dim/config.json"
    $output_file = "outputs/week2/wwprd_latent${dim}_results.json"

    if (-not (Test-Path $model_path)) {
        Write-Host "Model not found: $model_path" -ForegroundColor Yellow
        Write-Host "Skipping latent_dim=$dim" -ForegroundColor Yellow
        continue
    }

    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Evaluating model with latent_dim=$dim" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Cyan

    python scripts/evaluate_compression.py `
        --model_path $model_path `
        --config_path $config_path `
        --compression_ratios 4 8 16 32 `
        --quantization_bits $quantization_bits `
        --output_file $output_file `
        --num_test_samples 500 `
        --device auto

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Evaluation failed: latent_dim=$dim" -ForegroundColor Red
        continue
    }

    Write-Host "Completed: latent_dim=$dim" -ForegroundColor Green
    Write-Host ""
}

Write-Host "All models evaluation completed!" -ForegroundColor Green
Write-Host "Results saved in: outputs/week2/" -ForegroundColor Cyan
