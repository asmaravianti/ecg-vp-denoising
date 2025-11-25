# 使用latent_dim=2训练 - 目标更高CR和QS

Write-Host "🚀 训练latent_dim=2模型 - 目标CR≈22:1" -ForegroundColor Cyan
Write-Host ""

python scripts/train_mitbih.py `
    --loss_type wwprd `
    --latent_dim 2 `
    --num_records 20 `
    --epochs 200 `
    --quantization_aware `
    --quantization_bits 4 `
    --qat_probability 0.5 `
    --save_model `
    --output_dir outputs/wwprd_latent2_qat

Write-Host ""
Write-Host "✅ 训练完成！评估模型..." -ForegroundColor Green

# 评估
python scripts/evaluate_compression.py `
    --model_path outputs/wwprd_latent2_qat/best_model.pth `
    --config_path outputs/wwprd_latent2_qat/config.json `
    --quantization_bits 4 `
    --compression_ratios 16 20 24 32 `
    --num_test_samples 500 `
    --output_file outputs/wwprd_latent2_qat/qat_results.json

Write-Host ""
Write-Host "📊 计算QS..." -ForegroundColor Cyan
Write-Host "如果CR≈22, PRD<44%, 则QS>0.5 ✅" -ForegroundColor Yellow

