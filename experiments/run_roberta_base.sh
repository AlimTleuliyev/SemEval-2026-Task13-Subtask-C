#!/bin/bash

echo "================================================="
echo "Training RoBERTa from Scratch on Full Dataset (900K)"
echo "================================================="

python train.py \
    --model_name roberta-scratch \
    --model_path roberta-base \
    --train_size 900000 \
    --val_size 200000 \
    --max_length 510 \
    --num_epochs 5 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 4 \
    --output_dir ./models/roberta_scratch_full \
    --logging_steps 50 \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 1000 \
    --do_train \
    --do_eval \
    --do_predict \
    --submission_file ./predictions/roberta_scratch_submission.csv

echo ""
echo "✅ RoBERTa from scratch training complete!"
echo "Model saved to: ./models/roberta_scratch_full/final"
echo "Submission saved to: ./predictions/roberta_scratch_submission.csv"
