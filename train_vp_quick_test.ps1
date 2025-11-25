# Quick VP Test (10 records, 20 epochs - ~15-20 minutes)
# This will quickly verify VP model works and give us preliminary results

Write-Host "Quick VP Test (10 records, 20 epochs)..."
python scripts/train_mitbih.py --num_records 10 --epochs 20 --batch_size 32 --loss_type wwprd --weight_alpha 2.0 --latent_dim 2 --quantization_aware --qat_probability 0.7 --model_type vp --output_dir ./outputs/vp_quick_test

Write-Host "Quick test complete! Check outputs/vp_quick_test/"




