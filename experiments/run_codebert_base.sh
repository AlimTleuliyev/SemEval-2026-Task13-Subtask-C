#!/bin/bash

echo "========================================="
echo "Training CodeBERT on Full Dataset (900K)"
echo "========================================="

python train.py \
    --model_name codebert \
    --model_path microsoft/codebert-base \
    --train_size 900000 \
    --val_size 200000 \
    --max_length 510 \
    --num_epochs 7 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 4 \
    --output_dir ./models/codebert_full \
    --logging_steps 50 \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 1000 \
    --do_train \
    --do_eval \
    --do_predict \
    --submission_file ./predictions/codebert_submission.csv

echo ""
echo "✅ CodeBERT training complete!"
echo "Model saved to: ./models/codebert_full/final"
echo "Submission saved to: ./predictions/codebert_submission.csv"
