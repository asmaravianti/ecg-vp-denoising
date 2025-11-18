"""计算QS和QSN分数，并生成对比表格。

从压缩比评估结果中计算QS (CR/PRD) 和 QSN (CR/PRDN)，
生成表格用于论文。
"""

import json
from pathlib import Path
from typing import Dict, List

def load_results(results_file: str) -> Dict:
    """加载压缩比评估结果。"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_qs_scores(results: Dict) -> List[Dict]:
    """计算QS和QSN分数。"""
    table_rows = []

    for cr_key in sorted(results.keys(), key=int):
        m = results[cr_key]
        cr = m.get("actual_cr", float(cr_key))  # 使用实际CR
        prd = m["PRD"]      # PRD百分比
        prdn = m.get("PRDN", prd)  # PRDN百分比
        wwprd = m.get("WWPRD", 0)
        snr_imp = m.get("SNR_improvement", 0)

        # 计算QS和QSN
        qs = cr / prd if prd > 0 else 0.0
        qsn = cr / prdn if prdn > 0 else 0.0

        table_rows.append({
            "CR": round(cr, 2),
            "PRD": round(prd, 2),
            "PRDN": round(prdn, 2),
            "WWPRD": round(wwprd, 2),
            "SNR_imp": round(snr_imp, 2),
            "QS": round(qs, 4),
            "QSN": round(qsn, 4)
        })

    return table_rows

def print_table(table_rows: List[Dict], model_name: str = "Model"):
    """打印表格到控制台。"""
    print("=" * 90)
    print(f"压缩比评估结果 - {model_name} - QS和QSN分数")
    print("=" * 90)
    print(f"{'CR':<8} {'PRD(%)':<10} {'PRDN(%)':<10} {'WWPRD(%)':<12} {'SNR(dB)':<10} {'QS':<12} {'QSN':<12}")
    print("-" * 90)

    for row in table_rows:
        print(f"{row['CR']:<8.2f} {row['PRD']:<10.2f} {row['PRDN']:<10.2f} "
              f"{row['WWPRD']:<12.2f} {row['SNR_imp']:<10.2f} "
              f"{row['QS']:<12.4f} {row['QSN']:<12.4f}")

def generate_latex_table(table_rows: List[Dict], model_name: str = "WWPRD") -> str:
    """生成LaTeX表格代码。"""
    latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{压缩比评估结果：{model_name}模型的QS和QSN分数}}
\\label{{tab:compression_results_{model_name.lower()}}}
\\begin{{tabular}}{{ccccccc}}
\\toprule
CR & PRD (\\%) & PRDN (\\%) & WWPRD (\\%) & SNR (dB) & QS & QSN \\\\
\\midrule
"""

    for row in table_rows:
        latex += (f"{row['CR']:.2f} & {row['PRD']:.2f} & {row['PRDN']:.2f} & "
                 f"{row['WWPRD']:.2f} & {row['SNR_imp']:.2f} & "
                 f"{row['QS']:.4f} & {row['QSN']:.4f} \\\\\n")

    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex

def main():
    """主函数：处理所有模型的结果。"""
    import argparse

    parser = argparse.ArgumentParser(description="计算QS和QSN分数")
    parser.add_argument(
        '--results_dir',
        type=str,
        default='outputs/week2',
        help='结果文件目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/week2',
        help='输出目录'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有结果文件
    result_files = list(results_dir.glob("*_results.json"))

    if not result_files:
        print(f"❌ 未找到结果文件在 {results_dir}")
        print("请先运行评估脚本：")
        print("  .\\evaluate_all_models.ps1")
        return

    all_tables = {}

    # 处理每个模型的结果
    for result_file in result_files:
        print(f"\n处理: {result_file.name}")

        # 从文件名提取模型名称
        model_name = result_file.stem.replace("_results", "")

        try:
            results = load_results(str(result_file))
            table_rows = calculate_qs_scores(results)

            # 打印表格
            print_table(table_rows, model_name)

            # 保存JSON
            output_json = {
                "model": model_name,
                "results": table_rows
            }
            json_path = output_dir / f"{model_name}_qs_table.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_json, f, indent=2, ensure_ascii=False)
            print(f"✓ JSON已保存: {json_path}")

            # 生成LaTeX表格
            latex_table = generate_latex_table(table_rows, model_name)
            tex_path = output_dir / f"{model_name}_qs_table.tex"
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_table)
            print(f"✓ LaTeX表格已保存: {tex_path}")

            all_tables[model_name] = table_rows

        except Exception as e:
            print(f"❌ 处理失败 {result_file}: {e}")
            continue

    # 生成对比表格（如果有多个模型）
    if len(all_tables) > 1:
        print("\n" + "=" * 90)
        print("模型对比：QS分数")
        print("=" * 90)
        print(f"{'CR':<8} ", end="")
        for model_name in all_tables.keys():
            print(f"{model_name}-QS{'':<10}", end="")
        print("Winner")
        print("-" * 90)

        # 获取所有CR值
        all_crs = set()
        for table in all_tables.values():
            all_crs.update([row['CR'] for row in table])

        for cr in sorted(all_crs):
            print(f"{cr:<8.2f} ", end="")
            best_qs = -1
            best_model = ""
            for model_name, table in all_tables.items():
                row = next((r for r in table if abs(r['CR'] - cr) < 0.1), None)
                if row:
                    qs = row['QS']
                    print(f"{qs:<15.4f}", end="")
                    if qs > best_qs:
                        best_qs = qs
                        best_model = model_name
                else:
                    print(f"{'N/A':<15}", end="")
            print(best_model)

    print("\n" + "=" * 90)
    print("✓ 所有表格生成完成！")
    print(f"输出目录: {output_dir}")
    print("=" * 90)

if __name__ == "__main__":
    main()

