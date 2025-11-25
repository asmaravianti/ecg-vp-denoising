"""Summarize best scores for every model QS table under outputs/week2."""
import json
from pathlib import Path
from typing import Dict, List

RESULTS_DIR = Path("outputs/week2")
qs_files = sorted(RESULTS_DIR.glob("*_qs_table.json"))

if not qs_files:
    raise SystemExit("No *_qs_table.json files found in outputs/week2/")

rows: List[Dict[str, float]] = []

print("=" * 100)
print("SUMMARY TABLE - Best Performance by Model")
print("=" * 100)
print(
    f"{'Model':<30} {'CR':<8} {'PRD(%)':<10} {'PRDN(%)':<10} "
    f"{'WWPRD(%)':<12} {'QS':<12} {'QSN':<12}"
)
print("-" * 100)

for file_path in qs_files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", file_path.stem.replace("_qs_table", ""))
    results = data.get("results", [])
    if not results:
        continue

    best_entry = max(results, key=lambda r: r.get("QS", 0.0))

    row = {
        "model": model,
        "CR": best_entry["CR"],
        "PRD": best_entry["PRD"],
        "PRDN": best_entry["PRDN"],
        "WWPRD": best_entry["WWPRD"],
        "QS": best_entry["QS"],
        "QSN": best_entry["QSN"],
    }
    rows.append(row)

    print(
        f"{model:<30} {row['CR']:<8.2f} {row['PRD']:<10.2f} {row['PRDN']:<10.2f} "
        f"{row['WWPRD']:<12.2f} {row['QS']:<12.4f} {row['QSN']:<12.4f}"
    )

print("=" * 100)

if rows:
    top_qs = max(rows, key=lambda r: r["QS"])
    top_cr = max(rows, key=lambda r: r["CR"])

    print("\nKey Findings:")
    print(
        f"- Best QS: {top_qs['model']} (CR={top_qs['CR']:.2f}:1, QS={top_qs['QS']:.3f}, "
        f"QSN={top_qs['QSN']:.3f})"
    )
    print(
        f"- Highest CR achieved: {top_cr['model']} (CR={top_cr['CR']:.2f}:1, "
        f"QS={top_cr['QS']:.3f})"
    )

print("\nAll tables saved in: outputs/week2/")
print("  - *_qs_table.json (JSON format)")
print("  - *_qs_table.tex (LaTeX format)")
print("  - final_comparison_table.tex (Summary table)")

