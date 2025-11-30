#!/bin/bash

echo "=========================================================="
echo "Training OPTIMIZED UniXcoder with Class Weights (900K)"
echo "=========================================================="
echo ""
echo "Optimizations:"
echo "  - Class weights: [0.4635, 1.0690, 2.6310, 1.8983]"
echo "  - 10 epochs (vs 7 baseline)"
echo "  - Learning rate: 2e-5 (proven effective)"
echo "  - Target: F1 0.87-0.88 (vs 0.8598 baseline)"
echo ""
echo "Expected improvements:"
echo "  - Hybrid class F1: 0.77 -> 0.80+"
echo "  - Overall F1: 0.8598 -> 0.87-0.88"
echo ""
echo "=========================================================="

python train.py \
    --model_name unixcoder \
    --model_path microsoft/unixcoder-base \
    --train_size 900000 \
    --val_size 200000 \
    --max_length 510 \
    --num_epochs 10 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 4 \
    --class_weights "0.4635,1.0690,2.6310,1.8983" \
    --output_dir ./models/unixcoder_optimized \
    --logging_steps 50 \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 1000 \
    --do_train \
    --do_eval \
    --do_predict \
    --submission_file ./predictions/unixcoder_optimized_submission.csv

echo ""
echo "✅ Optimized UniXcoder training complete!"
echo ""
echo "Results saved to:"
echo "  - Model: ./models/unixcoder_optimized/final"
echo "  - Submission: ./predictions/unixcoder_optimized_submission.csv"
echo ""
echo "Class weight breakdown:"
echo "  - Human (53.9%):      weight = 0.4635"
echo "  - AI (23.4%):         weight = 1.0690"
echo "  - Hybrid (9.5%):      weight = 2.6310 (highest)"
echo "  - Adversarial (13.2%): weight = 1.8983"
echo ""
