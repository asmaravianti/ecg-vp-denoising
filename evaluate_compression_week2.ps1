# PowerShell script for running Week 2 compression evaluation
# Usage: .\evaluate_compression_week2.ps1

python scripts/evaluate_compression.py `
    --model_path outputs/week1_presentation/best_model.pth `
    --config_path outputs/week1_presentation/config.json `
    --compression_ratios 4 8 16 32 `
    --quantization_bits 8 `
    --num_test_samples 500 `
    --output_file outputs/week2/cr_sweep_results.json

Write-Host "`n✅ Compression evaluation complete!" -ForegroundColor Green
Write-Host "📊 Results saved to: outputs/week2/cr_sweep_results.json" -ForegroundColor Cyan
Write-Host "`nNext: Run visualization:" -ForegroundColor Yellow
Write-Host "  python scripts/plot_rate_distortion.py --results_file outputs/week2/cr_sweep_results.json --output_dir outputs/week2_final" -ForegroundColor White

