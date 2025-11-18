# PowerShell script to calculate Quality Scores (QS)
# Usage: .\calculate_qs.ps1

$env:PYTHONPATH = "."

python scripts/calculate_qs_scores.py --wwprd_model outputs/loss_comparison_wwprd/best_model.pth --wwprd_config outputs/loss_comparison_wwprd/config.json --combined_model outputs/loss_comparison_combined_alpha0.5/best_model.pth --combined_config outputs/loss_comparison_combined_alpha0.5/config.json --record_ids 117 119 --quantization_bits 8 --output outputs/qs_analysis.json

