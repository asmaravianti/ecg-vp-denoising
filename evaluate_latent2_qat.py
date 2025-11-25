"""评估latent_dim=2的QAT模型并生成QS table"""

import subprocess
import json
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("📊 评估latent_dim=2 QAT模型")
    print("="*70)

    model_path = "outputs/wwprd_latent2_qat_optimized/best_model.pth"
    config_path = "outputs/wwprd_latent2_qat_optimized/config.json"
    results_file = "outputs/wwprd_latent2_qat_optimized/qat_compression_results.json"

    # 步骤1: 评估压缩性能
    print("\n🔍 步骤1: 评估压缩性能（post-quantization）...")
    print("   这很重要 - 会显示真实的QS值")

    cmd = [
        "python", "scripts/evaluate_compression.py",
        "--model_path", model_path,
        "--config_path", config_path,
        "--quantization_bits", "4",
        "--compression_ratios", "16", "20", "24", "32",
        "--num_test_samples", "500",
        "--output_file", results_file
    ]

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print("❌ 评估失败！")
        return

    print("✅ 压缩评估完成！")

    # 步骤2: 读取结果并生成QS table
    print("\n📋 步骤2: 生成QS Table...")

    if not Path(results_file).exists():
        print(f"❌ 结果文件不存在: {results_file}")
        return

    with open(results_file, 'r') as f:
        results = json.load(f)

    # 转换为QS table格式
    qs_results = []
    for key, metrics in results.items():
        if isinstance(metrics, dict) and 'PRD' in metrics:
            cr = metrics.get('actual_cr', 0)
            prd = metrics.get('PRD', 0)
            prdn = metrics.get('PRDN', 0)
            wwprd = metrics.get('WWPRD', 0)

            # QS = CR / PRD
            qs = cr / prd if prd > 0 else 0
            qsn = cr / prdn if prdn > 0 else 0

            qs_results.append({
                "CR": round(cr, 2),
                "PRD": round(prd, 2),
                "PRDN": round(prdn, 2),
                "WWPRD": round(wwprd, 2),
                "SNR_imp": round(metrics.get('SNR_improvement', 0), 2),
                "QS": round(qs, 4),
                "QSN": round(qsn, 4)
            })

    # 保存QS table
    qs_table_path = Path("outputs/week2/wwprd_latent2_qat_qs_table.json")
    qs_table_path.parent.mkdir(parents=True, exist_ok=True)

    qs_table_data = {
        "model": "wwprd_latent2_qat",
        "results": qs_results
    }

    with open(qs_table_path, 'w') as f:
        json.dump(qs_table_data, f, indent=2)

    # 生成LaTeX table
    tex_path = qs_table_path.with_suffix('.tex')
    best = max(qs_results, key=lambda x: x['QS'])

    tex_content = f"""\\begin{{table}}[h]
\\centering
\\caption{{Latent Dimension 2 QAT Model: QS and QSN Scores}}
\\label{{tab:latent2_qat_qs}}
\\small
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Quantization}} & \\textbf{{CR}} & \\textbf{{PRD (\\%)}} & \\textbf{{PRDN (\\%)}} & \\textbf{{WWPRD (\\%)}} & \\textbf{{QS}} & \\textbf{{QSN}} \\\\
\\midrule
"""

    for r in sorted(qs_results, key=lambda x: x['CR']):
        marker = "\\textbf{" if r['QS'] == best['QS'] else ""
        marker_end = "}" if r['QS'] == best['QS'] else ""
        tex_content += f"{r['CR']:.1f} & {r['CR']:.2f} & {r['PRD']:.2f} & {r['PRDN']:.2f} & {r['WWPRD']:.2f} & {marker}{r['QS']:.4f}{marker_end} & {r['QSN']:.4f} \\\\\n"

    tex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    with open(tex_path, 'w') as f:
        f.write(tex_content)

    # 打印结果
    print("\n" + "="*80)
    print("📊 Latent Dimension 2 QAT模型 QS Table 结果")
    print("="*80)
    print(f"{'CR':<10} {'PRD (%)':<12} {'PRDN (%)':<12} {'WWPRD (%)':<14} {'QS':<10} {'QSN':<10}")
    print("-"*80)

    for r in sorted(qs_results, key=lambda x: x['QS'], reverse=True):
        best_marker = " ⭐" if r['QS'] == best['QS'] else ""
        target_marker = " ✅" if r['QS'] >= 0.5 else ""
        print(f"{r['CR']:<10.2f} {r['PRD']:<12.2f} {r['PRDN']:<12.2f} {r['WWPRD']:<14.2f} {r['QS']:<10.4f} {r['QSN']:<10.4f}{best_marker}{target_marker}")

    print("="*80)
    print(f"\n🏆 最佳QS: {best['QS']:.4f} (CR={best['CR']:.2f}, PRD={best['PRD']:.2f}%)")

    # 检查是否达到目标
    if best['QS'] >= 0.5:
        print(f"\n🎉 恭喜！达到目标 QS > 0.5 ✅")
        print(f"   目标: QS > 0.5")
        print(f"   实际: QS = {best['QS']:.4f}")
    else:
        print(f"\n⚠️  未达到目标 QS > 0.5")
        print(f"   目标: QS > 0.5")
        print(f"   实际: QS = {best['QS']:.4f}")
        print(f"   差距: {0.5 - best['QS']:.4f}")

        # 建议
        if best['QS'] >= 0.45:
            print(f"\n💡 建议: 非常接近！可以尝试:")
            print(f"   1. 使用3-bit量化 (CR会更高)")
            print(f"   2. 延长训练到250 epochs")
            print(f"   3. 调整QAT概率到0.8")
        elif best['QS'] >= 0.4:
            print(f"\n💡 建议: 接近目标，可以:")
            print(f"   1. 使用3-bit量化")
            print(f"   2. 尝试latent_dim=1 (CR会更高)")

    # 与baseline对比
    baseline_file = Path("outputs/week2/wwprd_latent4_qs_table.json")
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
            baseline_results = baseline_data.get('results', [])
            if baseline_results:
                baseline_best = max(baseline_results, key=lambda x: x.get('QS', 0))
                baseline_qs = baseline_best.get('QS', 0)
                improvement = ((best['QS'] - baseline_qs) / baseline_qs * 100) if baseline_qs > 0 else 0

                print(f"\n📈 与Baseline (latent4) 对比:")
                print(f"   Baseline最佳QS: {baseline_qs:.4f} (CR={baseline_best.get('CR', 0):.2f}, PRD={baseline_best.get('PRD', 0):.2f}%)")
                print(f"   Latent2 QAT最佳QS: {best['QS']:.4f} (CR={best['CR']:.2f}, PRD={best['PRD']:.2f}%)")
                print(f"   提升: {improvement:+.2f}%")

    print(f"\n✅ QS Table已保存:")
    print(f"   JSON: {qs_table_path}")
    print(f"   LaTeX: {tex_path}")

    # 更新可视化表格
    try:
        print("\n🖼️  更新可视化QS表格...")
        subprocess.run(["python", "scripts/render_qs_summary.py"], check=False)
        print("✅ 可视化表格已更新")
    except:
        print("⚠️  可视化表格更新失败（可选）")

if __name__ == "__main__":
    main()

