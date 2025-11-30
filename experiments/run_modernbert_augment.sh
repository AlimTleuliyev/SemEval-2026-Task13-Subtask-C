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
    --max_length 510 \
    --num_epochs 3 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 4 \
    --use_random_crop \
    --output_dir ./models/modernbert_augmentation_full \
    --logging_steps 100 \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 1000 \
    --do_train \
    --do_eval \
    --do_predict \
    --submission_file ./predictions/modernbert_augmentation_submission.csv

echo ""
echo "✅ ModernBERT-Base training complete!"
echo "Model saved to: ./models/modernbert_augmentation_full/final"
echo "Submission saved to: ./predictions/modernbert_augmentation_submission.csv"
echo ""
echo "📊 Next steps:"
echo "  1. Check results in findings.md"
echo "  2. Compare with UniXcoder (baseline: 0.8598)"
echo "  3. Try ensembling ModernBERT + UniXcoder for best results"
