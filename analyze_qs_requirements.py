"""分析QS要求：要达到QS > 1需要什么条件"""
import json
from pathlib import Path

print("=" * 80)
print("QS REQUIREMENT ANALYSIS")
print("=" * 80)
print("\nQS = CR / PRD")
print("For QS > 1, we need: CR > PRD (in percentage)\n")

print("Current Results:")
print("-" * 80)
print(f"{'Model':<20} {'CR':<8} {'PRD(%)':<10} {'QS':<10} {'Status':<15}")
print("-" * 80)

files = [
    ('wwprd_latent8', 'outputs/week2/wwprd_latent8_qs_table.json'),
    ('wwprd_latent16', 'outputs/week2/wwprd_latent16_qs_table.json'),
    ('wwprd_latent32', 'outputs/week2/wwprd_latent32_qs_table.json')
]

for model_name, file_path in files:
    data = json.load(open(file_path))
    results = data['results']

    # Find best QS
    best = max(results, key=lambda x: x['QS'])
    cr = best['CR']
    prd = best['PRD']
    qs = best['QS']

    status = "❌ QS < 1" if qs < 1 else "✅ QS ≥ 1"
    print(f"{model_name:<20} {cr:<8.2f} {prd:<10.2f} {qs:<10.4f} {status:<15}")

print("\n" + "=" * 80)
print("REQUIREMENTS TO ACHIEVE QS > 1")
print("=" * 80)

print("\nOption 1: Improve PRD (Lower is Better)")
print("-" * 80)
print("Target: PRD < 5% (Excellent clinical quality)")
print("Current: PRD ≈ 35% (Too high!)")
print("\nTo achieve QS > 1 with current CR=5.5:")
print("  Need: PRD < 5.5%")
print("  Gap: Need to reduce PRD from 35% to 5% (7x improvement)")
print("\nActions:")
print("  1. Train longer (100-200 epochs instead of 50)")
print("  2. Use better architecture (deeper network)")
print("  3. Improve training data (more records, better augmentation)")
print("  4. Fine-tune hyperparameters (learning rate, weight decay)")

print("\nOption 2: Increase CR (Higher is Better)")
print("-" * 80)
print("Target: CR > 20:1 (High compression)")
print("Current: CR ≈ 5.5:1 (Low compression)")
print("\nTo achieve QS > 1 with current PRD=35%:")
print("  Need: CR > 35:1")
print("  Gap: Need to increase CR from 5.5 to 35 (6.4x improvement)")
print("\nActions:")
print("  1. Use smaller latent_dim (4 instead of 8)")
print("  2. Use lower quantization (2-bit instead of 4-bit)")
print("  3. Increase window length (more samples per window)")
print("  4. Apply entropy coding (Huffman, run-length)")

print("\n" + "=" * 80)
print("RECOMMENDED APPROACH: Combined Strategy")
print("=" * 80)
print("\nBest path to QS > 1:")
print("  1. Train model with latent_dim=4 (target CR ≈ 10-15:1)")
print("  2. Train for 100+ epochs to reduce PRD to < 10%")
print("  3. Use 4-bit quantization")
print("  4. Expected: CR=12:1, PRD=8% → QS = 12/8 = 1.5 ✅")

print("\n" + "=" * 80)
print("COMPARISON WITH LITERATURE (Kovács et al. 2020)")
print("=" * 80)
print("\nFrom Table V in the paper:")
print("  - Best methods: QS = 2-5 (CR=10-20:1, PRD=2-5%)")
print("  - Good methods: QS = 1-2 (CR=8-15:1, PRD=4-8%)")
print("  - Acceptable: QS = 0.5-1 (CR=5-10:1, PRD=5-10%)")
print("\nYour current: QS ≈ 0.15 (needs significant improvement)")

print("\n" + "=" * 80)



