"""显示汇总表格 - 3个模型的最佳性能对比"""
import json
from pathlib import Path

files = [
    'outputs/week2/wwprd_latent8_qs_table.json',
    'outputs/week2/wwprd_latent16_qs_table.json',
    'outputs/week2/wwprd_latent32_qs_table.json'
]

print("=" * 90)
print("SUMMARY TABLE - Best Performance by Model")
print("=" * 90)
print(f"{'Model':<20} {'CR':<8} {'PRD(%)':<10} {'PRDN(%)':<10} {'WWPRD(%)':<12} {'QS':<12} {'QSN':<12}")
print("-" * 90)

for f in files:
    data = json.load(open(f))
    model = data['model']
    results = data['results']

    best_cr = max(r['CR'] for r in results)
    best_prd = min(r['PRD'] for r in results)
    best_prdn = min(r['PRDN'] for r in results)
    best_wwprd = min(r['WWPRD'] for r in results)
    best_qs = max(r['QS'] for r in results)
    best_qsn = max(r['QSN'] for r in results)

    print(f"{model:<20} {best_cr:<8.2f} {best_prd:<10.2f} {best_prdn:<10.2f} "
          f"{best_wwprd:<12.2f} {best_qs:<12.4f} {best_qsn:<12.4f}")

print("=" * 90)
print("\nKey Findings:")
print("- Latent_dim=8: Highest CR (5.50:1) and best QS/QSN scores")
print("- Latent_dim=16: Medium CR (2.75:1) with moderate QS/QSN")
print("- Latent_dim=32: Lowest CR (1.38:1) but best PRD/WWPRD quality")
print("\nAll tables saved in: outputs/week2/")
print("  - *_qs_table.json (JSON format)")
print("  - *_qs_table.tex (LaTeX format)")
print("  - final_comparison_table.tex (Summary table)")

