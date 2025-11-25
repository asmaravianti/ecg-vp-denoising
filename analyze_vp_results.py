"""综合分析所有VP实验结果"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

def load_experiment_results(exp_dir: str) -> Optional[Dict]:
    """加载单个实验的结果"""
    exp_path = Path(f"outputs/{exp_dir}")

    if not exp_path.exists():
        return None

    results = {
        "experiment": exp_dir,
        "has_model": (exp_path / "best_model.pth").exists(),
        "has_metrics": (exp_path / "final_metrics.json").exists(),
        "has_config": (exp_path / "config.json").exists(),
        "has_compression": (exp_path / "compression_results.json").exists() or (exp_path / "qat_compression_results.json").exists(),
    }

    # 加载配置
    if results["has_config"]:
        with open(exp_path / "config.json", 'r') as f:
            config = json.load(f)
            results.update({
                "num_records": config.get("num_records", "N/A"),
                "epochs": config.get("epochs", "N/A"),
                "latent_dim": config.get("latent_dim", "N/A"),
                "qat": config.get("quantization_aware", False),
                "qat_prob": config.get("qat_probability", "N/A"),
                "model_type": config.get("model_type", "N/A"),
            })

    # 加载训练指标
    if results["has_metrics"]:
        with open(exp_path / "final_metrics.json", 'r') as f:
            metrics = json.load(f)
            results.update({
                "PRD": metrics.get("PRD", None),
                "PRD_std": metrics.get("PRD_std", None),
                "WWPRD": metrics.get("WWPRD", None),
                "WWPRD_std": metrics.get("WWPRD_std", None),
                "SNR_improvement": metrics.get("SNR_improvement", None),
                "SNR_out": metrics.get("SNR_out", None),
                "val_loss": metrics.get("val_loss", None),
            })

    # 加载压缩评估结果
    compression_file = None
    if (exp_path / "compression_results.json").exists():
        compression_file = exp_path / "compression_results.json"
    elif (exp_path / "qat_compression_results.json").exists():
        compression_file = exp_path / "qat_compression_results.json"

    if compression_file:
        with open(compression_file, 'r') as f:
            comp_results = json.load(f)
            # 提取最佳QS
            if isinstance(comp_results, list) and len(comp_results) > 0:
                best_result = max(comp_results, key=lambda x: x.get("QS", 0))
                results.update({
                    "postQ_PRD": best_result.get("PRD", None),
                    "postQ_WWPRD": best_result.get("WWPRD", None),
                    "CR": best_result.get("compression_ratio", None),
                    "QS": best_result.get("QS", None),
                    "QSN": best_result.get("QSN", None),
                })
            elif isinstance(comp_results, dict):
                results.update({
                    "postQ_PRD": comp_results.get("PRD", None),
                    "postQ_WWPRD": comp_results.get("WWPRD", None),
                    "CR": comp_results.get("compression_ratio", None),
                    "QS": comp_results.get("QS", None),
                })

    # 检查估算结果
    if (exp_path / "estimated_results.json").exists():
        with open(exp_path / "estimated_results.json", 'r') as f:
            est = json.load(f)
            if "estimated_qs" in est:
                results["estimated_QS"] = est["estimated_qs"].get("QS", None)
                results["estimated_postQ_PRD"] = est["estimated_post_quantization"].get("PRD", None)

    return results

def analyze_vp_experiments():
    """分析所有VP实验"""
    vp_experiments = [
        "vp_dry_run",
        "vp_experiment_latent2",
        "vp_final",
        "vp_quick_test",
        "vp_smoke_test",
    ]

    all_results = []
    for exp in vp_experiments:
        result = load_experiment_results(exp)
        if result:
            all_results.append(result)

    # 创建DataFrame
    df_data = []
    for r in all_results:
        df_data.append({
            "实验": r["experiment"],
            "记录数": r.get("num_records", "N/A"),
            "Epochs": r.get("epochs", "N/A"),
            "Latent Dim": r.get("latent_dim", "N/A"),
            "QAT": "✓" if r.get("qat") else "✗",
            "Clean PRD (%)": f"{r.get('PRD', 0):.2f}" if r.get("PRD") else "N/A",
            "Clean WWPRD (%)": f"{r.get('WWPRD', 0):.2f}" if r.get("WWPRD") else "N/A",
            "SNR改善 (dB)": f"{r.get('SNR_improvement', 0):.2f}" if r.get("SNR_improvement") else "N/A",
            "Post-Q PRD (%)": f"{r.get('postQ_PRD', 0):.2f}" if r.get("postQ_PRD") else "N/A",
            "CR": f"{r.get('CR', 0):.2f}" if r.get("CR") else "N/A",
            "QS": f"{r.get('QS', 0):.4f}" if r.get("QS") else (f"{r.get('estimated_QS', 0):.4f}*" if r.get("estimated_QS") else "N/A"),
            "有模型": "✓" if r.get("has_model") else "✗",
            "有压缩评估": "✓" if r.get("has_compression") else "✗",
        })

    df = pd.DataFrame(df_data)

    print("=" * 100)
    print("VP实验结果综合分析")
    print("=" * 100)
    print("\n")
    print(df.to_string(index=False))
    print("\n")
    print("说明:")
    print("  * = 估算值（基于quantization gap）")
    print("  QAT = Quantization-Aware Training")
    print("  Post-Q PRD = Post-Quantization PRD（实际压缩后的PRD）")
    print("  QS = Quality Score = CR / (PRD/100)")
    print("\n")

    # 找出最佳结果
    print("=" * 100)
    print("最佳结果分析")
    print("=" * 100)

    # 最佳Clean PRD
    best_clean = min([r for r in all_results if r.get("PRD")], key=lambda x: x.get("PRD", float('inf')))
    print(f"\n✅ 最佳Clean PRD: {best_clean['experiment']}")
    print(f"   PRD: {best_clean.get('PRD', 0):.2f}%")
    print(f"   WWPRD: {best_clean.get('WWPRD', 0):.2f}%")
    print(f"   SNR改善: {best_clean.get('SNR_improvement', 0):.2f} dB")

    # 最佳QS（如果有）
    results_with_qs = [r for r in all_results if r.get("QS") or r.get("estimated_QS")]
    if results_with_qs:
        best_qs = max(results_with_qs, key=lambda x: x.get("QS") or x.get("estimated_QS", 0))
        qs_val = best_qs.get("QS") or best_qs.get("estimated_QS")
        print(f"\n✅ 最佳QS: {best_qs['experiment']}")
        print(f"   QS: {qs_val:.4f}" + (" (估算)" if best_qs.get("estimated_QS") else ""))
        if best_qs.get("CR"):
            print(f"   CR: {best_qs.get('CR', 0):.2f}:1")
        if best_qs.get("postQ_PRD"):
            print(f"   Post-Q PRD: {best_qs.get('postQ_PRD', 0):.2f}%")

    # 与Baseline对比
    print("\n" + "=" * 100)
    print("与Baseline对比 (wwprd_latent2_qat_optimized)")
    print("=" * 100)

    baseline_path = Path("outputs/wwprd_latent2_qat_optimized/final_metrics.json")
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        baseline_qs = 0.6078  # 已知值
        baseline_prd = baseline.get("PRD", 0)
        baseline_wwprd = baseline.get("WWPRD", 0)

        print(f"\nBaseline (标准卷积):")
        print(f"  Clean PRD: {baseline_prd:.2f}%")
        print(f"  Clean WWPRD: {baseline_wwprd:.2f}%")
        print(f"  QS: {baseline_qs:.4f}")

        if best_clean.get("PRD"):
            prd_improvement = ((baseline_prd - best_clean.get("PRD", 0)) / baseline_prd) * 100
            wwprd_improvement = ((baseline_wwprd - best_clean.get("WWPRD", 0)) / baseline_wwprd) * 100
            print(f"\nVP Layer最佳结果 ({best_clean['experiment']}):")
            print(f"  Clean PRD: {best_clean.get('PRD', 0):.2f}% (改善 {prd_improvement:.1f}%)")
            print(f"  Clean WWPRD: {best_clean.get('WWPRD', 0):.2f}% (改善 {wwprd_improvement:.1f}%)")

            if results_with_qs:
                qs_improvement = ((qs_val - baseline_qs) / baseline_qs) * 100
                print(f"  QS: {qs_val:.4f} (改善 {qs_improvement:.1f}%)")

    # 保存结果
    output_path = Path("outputs/vp_analysis_summary.json")
    summary = {
        "experiments": all_results,
        "best_clean_prd": best_clean["experiment"] if best_clean else None,
        "best_qs": best_qs["experiment"] if results_with_qs else None,
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ 分析结果已保存到: {output_path}")

    # 生成CSV
    csv_path = Path("outputs/vp_analysis_summary.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ CSV表格已保存到: {csv_path}")

if __name__ == "__main__":
    analyze_vp_experiments()
