# 优化训练脚本 - 目标QS > 0.5

Write-Host "🚀 开始优化训练 - 目标QS > 0.5" -ForegroundColor Cyan
Write-Host ""

# 方案1: 延长训练 + 更多数据 (推荐)
Write-Host "方案1: 延长训练到200 epochs + 使用48 records" -ForegroundColor Yellow
python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 4 `
    --num_records 48 `
    --epochs 200 `
    --quantization_aware `
    --quantization_bits 4 `
    --qat_probability 0.7 `
    --qat_mode ste `
    --lr 0.0005 `
    --save_model `
    --output_dir outputs/wwprd_latent4_qat_optimized

Write-Host ""
Write-Host "✅ 训练完成！现在评估模型..." -ForegroundColor Green

# 评估模型
python scripts/evaluate_compression.py `
    --model_path outputs/wwprd_latent4_qat_optimized/best_model.pth `
    --config_path outputs/wwprd_latent4_qat_optimized/config.json `
    --quantization_bits 4 `
    --compression_ratios 4 8 16 32 `
    --num_test_samples 500 `
    --output_file outputs/wwprd_latent4_qat_optimized/qat_results.json

Write-Host ""
Write-Host "📊 生成QS table..." -ForegroundColor Cyan
python fix_qat_qs_table.py

Write-Host ""
Write-Host "✅ 完成！检查 outputs/week2/wwprd_latent4_qat_optimized_qs_table.json" -ForegroundColor Green

