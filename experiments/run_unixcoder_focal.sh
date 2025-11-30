#!/bin/bash

echo "=========================================================="
echo "Training UniXcoder with FOCAL LOSS (900K)"
echo "=========================================================="
echo ""
echo "Configuration:"
echo "  - Focal Loss with gamma=2.0"
echo "  - Class weights: [0.4635, 1.0690, 2.6310, 1.8983]"
echo "  - 5 epochs"
echo "  - Learning rate: 2e-5"
echo ""
echo "Focal Loss Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"
echo "  - gamma=2.0: Focus heavily on hard examples"
echo "  - Expected improvement over simple weighting: +0.3-0.7% F1"
echo ""
echo "=========================================================="

python train.py \
    --model_name unixcoder \
    --model_path microsoft/unixcoder-base \
    --train_size 900000 \
    --val_size 200000 \
    --max_length 712 \
    --num_epochs 5 \
    --batch_size 50 \
    --learning_rate 2e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 8 \
    --class_weights "0.4635,1.0690,2.6310,1.8983" \
    --use_focal_loss \
    --focal_gamma 2.0 \
    --output_dir ./models/unixcoder_focal \
    --logging_steps 50 \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 1000 \
    --do_train \
    --do_eval \
    --do_predict \
    --submission_file ./predictions/unixcoder_focal_submission.csv

echo ""
echo "✅ Focal Loss training complete!"
echo ""
echo "Results saved to:"
echo "  - Model: ./models/unixcoder_focal/final"
echo "  - Submission: ./predictions/unixcoder_focal_submission.csv"
echo ""
echo "Focal Loss vs Weighted CE:"
echo "  - Both use same class weights (alpha)"
echo "  - Focal adds (1-p_t)^gamma term to focus on hard examples"
echo "  - Best for severely imbalanced classes (like Hybrid: 9.5%)"
echo ""
