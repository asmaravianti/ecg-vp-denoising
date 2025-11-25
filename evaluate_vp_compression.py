"""为VP模型运行压缩评估"""
import subprocess
import sys
from pathlib import Path

# 检查模型是否存在
model_path = Path("outputs/vp_final/best_model.pth")
config_path = Path("outputs/vp_final/config.json")

if not model_path.exists():
    print(f"❌ 模型文件不存在: {model_path}")
    sys.exit(1)

if not config_path.exists():
    print(f"❌ 配置文件不存在: {config_path}")
    sys.exit(1)

print("=" * 60)
print("运行VP模型压缩评估")
print("=" * 60)
print(f"模型: {model_path}")
print(f"配置: {config_path}")
print()

# 运行评估
cmd = [
    "python", "scripts/evaluate_compression.py",
    "--model_path", str(model_path),
    "--config_path", str(config_path),
    "--output_file", "outputs/vp_final/compression_results.json",
    "--quantization_bits", "4",
    "--num_test_samples", "100"
]

print(f"执行命令: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print()

if result.stderr:
    print("STDERR:")
    print(result.stderr)
    print()

if result.returncode == 0:
    print("✅ 压缩评估完成！")
    print(f"结果保存在: outputs/vp_final/compression_results.json")
else:
    print(f"❌ 评估失败，退出码: {result.returncode}")

