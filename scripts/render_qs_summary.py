"""
Generate an image of the Week 2 QS summary table.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("outputs/week2")
OUTPUT_PATH = RESULTS_DIR / "qs_summary_table.png"


def load_best_rows():
    rows = []
    for file_path in sorted(RESULTS_DIR.glob("*_qs_table.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        if not results:
            continue
        best = max(results, key=lambda r: r.get("QS", 0.0))
        rows.append(
            {
                "Model": data.get("model", file_path.stem.replace("_qs_table", "")),
                "CR (actual)": f"{best['CR']:.2f}:1",
                "PRD (%)": best["PRD"],
                "WWPRD (%)": best["WWPRD"],
                "QS": best["QS"],
                "QSN": best["QSN"],
            }
        )
    return rows


def render_table(rows):
    df = pd.DataFrame(rows).sort_values("QS", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 2 + 0.45 * len(df)))
    ax.axis("off")
    cell_text = []
    for _, row in df.iterrows():
        cell_text.append(
            [
                row["Model"],
                row["CR (actual)"],
                f"{row['PRD (%)']:.2f}",
                f"{row['WWPRD (%)']:.2f}",
                f"{row['QS']:.3f}",
                f"{row['QSN']:.3f}",
            ]
        )

    table = ax.table(
        cellText=cell_text,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    fig.suptitle("Week 2 Compression Summary (Best QS per Model)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved table image to {OUTPUT_PATH}")


def main():
    rows = load_best_rows()
    if not rows:
        raise SystemExit("No *_qs_table.json files with data.")
    render_table(rows)


if __name__ == "__main__":
    main()

