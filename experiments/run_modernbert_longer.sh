#!/bin/bash

echo "=========================================="
echo "Training ModernBERT-Base on Full Dataset"
echo "=========================================="
echo ""
echo "Model: answerdotai/ModernBERT-base"
echo "Context: Up to 8192 tokens (using 2048 for efficiency)"
echo "Features: Flash attention, rotary embeddings"
echo ""

python train.py \
    --model_name modernbert-base \
    --model_path answerdotai/ModernBERT-base \
    --train_size 900000 \
    --val_size 200000 \
    --max_length 1024 \
    --num_epochs 5 \
    --batch_size 24 \
    --learning_rate 2e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 12 \
    --output_dir ./models/modernbert_longer_full \
    --logging_steps 100 \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 1000 \
    --do_train \
    --do_eval \
    --do_predict \
    --submission_file ./predictions/modernbert_longer_submission.csv

echo ""
echo "✅ ModernBERT-Base training complete!"
echo "Model saved to: ./models/modernbert_longer_full/final"
echo "Submission saved to: ./predictions/modernbert_longer_submission.csv"
echo ""