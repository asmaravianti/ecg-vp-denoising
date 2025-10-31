# PowerShell script for running Week 2 visualization
# Usage: .\plot_week2_visualization.ps1

python scripts/plot_rate_distortion.py `
    --results_file outputs/week2/cr_sweep_results.json `
    --model_path outputs/week1_presentation/best_model.pth `
    --config_path outputs/week1_presentation/config.json `
    --output_dir outputs/week2_final

Write-Host "`n✅ Week 2 visualization complete!" -ForegroundColor Green
Write-Host "📊 Plots saved to: outputs/week2_final/" -ForegroundColor Cyan

