#!/bin/bash

echo "=========================================================================="
echo "Ensemble: UniXcoder + ModernBERT-Base"
echo "=========================================================================="
echo ""
echo "Model 1: UniXcoder (F1 0.8598)"
echo "Model 2: ModernBERT-Base (F1 0.8632)"
echo "Expected ensemble F1: 0.87-0.88"
echo ""

python3 ensemble.py \
    --model1_path ./models/unixcoder_full/final \
    --model1_tokenizer microsoft/unixcoder-base \
    --model1_type roberta \
    --model1_name UniXcoder \
    --model2_path ./models/modernbert_base_full/final \
    --model2_tokenizer answerdotai/ModernBERT-base \
    --model2_type modernbert \
    --model2_name ModernBERT-Base \
    --test_sample_file Task_C/test_sample.parquet \
    --test_file Task_C/test.parquet \
    --max_length 510 \
    --batch_size 32 \
    --output_file ./predictions/ensemble_unixcoder_modernbert.csv

echo ""
echo "✅ Ensemble complete!"
echo "Submission saved to: ./predictions/ensemble_unixcoder_modernbert.csv"
echo ""
echo "📊 Next steps:"
echo "  1. Check the F1 score in the output above"
echo "  2. Submit to Kaggle if F1 > 0.87"
echo "  3. Update findings.md with results"
