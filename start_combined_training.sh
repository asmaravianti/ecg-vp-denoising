#!/bin/bash
# Bash script to start Combined Loss training
# For teammate to run in parallel (Linux/Mac)

echo "========================================"
echo "Starting Combined Loss Model Training"
echo "========================================"
echo ""
echo "This will train the Combined Loss model (50 epochs)"
echo "Estimated time: 5-6 hours on CPU"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

echo ""
echo "Starting training..."
echo ""

python scripts/train_mitbih.py \
  --data_dir ./data/mitbih \
  --num_records 20 \
  --window_seconds 2.0 \
  --sample_rate 360 \
  --noise_type nstdb \
  --snr_db 10.0 \
  --nstdb_noise muscle_artifact \
  --model_type residual \
  --hidden_dims 32 64 128 \
  --latent_dim 32 \
  --loss_type combined \
  --combined_alpha 0.5 \
  --weight_alpha 2.0 \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.0005 \
  --weight_decay 0.0001 \
  --val_split 0.15 \
  --output_dir outputs/loss_comparison_combined_alpha0.5 \
  --save_model \
  --device auto

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Training Completed Successfully!"
    echo "========================================"
    echo "Results saved to: outputs/loss_comparison_combined_alpha0.5/"
else
    echo ""
    echo "========================================"
    echo "Training Failed!"
    echo "========================================"
    echo "Check the error messages above"
fi



