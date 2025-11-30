# AI-Generated Code Detection (SemEval-2026 Task 13, Subtask C)

This repository contains the code used to classify code snippets as Human, Machine-generated, Hybrid, or Adversarial for SemEval-2026 Task 13 (Subtask C). It includes transformer fine-tuning (UniXcoder, CodeBERT, RoBERTa scratch, ModernBERT) and classical TF-IDF + XGBoost baselines.

**Competition resources**
- Kaggle competition: https://www.kaggle.com/t/005ab8234f27424aa096b7c00a073722
- Task website: https://github.com/mbzuai-nlp/SemEval-2026-Task13/

## Repository Map (tracked)
- `train.py`: Transformer training/evaluation/prediction with optional class weights, focal loss, and random cropping.
- `eval.py`: Evaluate saved checkpoints on `test_sample.parquet` and generate `test.parquet` submissions.
- `eval.sh`: Convenience script to run `eval.py` across all saved checkpoints and write CSVs under `predictions/`.
- `ensemble.py`: Two-model ensembling with weighted averaging/voting and automatic weight search.
- `train_xgboost.py`: Character n-gram TF-IDF + XGBoost baseline (includes logging and submission generation).
- `experiments/*.sh`: Ready-made runs for the reported experiments (UniXcoder variants, CodeBERT, RoBERTa-scratch, ModernBERT base/longer/augmented, and ensembling).
- `predictions/*.csv`: Saved submissions produced by prior runs.
- `models/*`: Checkpoint directories (configs and trainer states are tracked; actual weights are not committed—train to regenerate).
- `notebooks/baseline.ipynb`, `notebooks/train_xgboost.ipynb`: Exploratory analysis for baselines.
- `requirements.txt`: Python dependencies.

## Setup
- Python ≥3.10; GPU strongly recommended for transformers.
- Install dependencies: `pip install -r requirements.txt`.
- Data: place the competition parquet files in `Task_C/` (`train.parquet`, `validation.parquet`, `test.parquet`, `test_sample.parquet`). These are not included in the repository.
- Hugging Face models download automatically; ensure access or pre-cache checkpoints if working offline.

## Training
Transformer training is handled by `train.py`. Example (ModernBERT base, full data):
```bash
python train.py \
  --model_name modernbert-base \
  --model_path answerdotai/ModernBERT-base \
  --train_size 900000 --val_size 200000 \
  --max_length 510 --num_epochs 3 --batch_size 64 \
  --learning_rate 2e-5 --weight_decay 0.001 --warmup_ratio 0.1 \
  --gradient_accumulation_steps 4 \
  --output_dir ./models/modernbert_base_full \
  --logging_steps 100 --eval_steps 1000 \
  --save_strategy steps --save_steps 1000 \
  --do_train --do_eval --do_predict \
  --submission_file ./predictions/modernbert_base_submission.csv
```
Preset runs for each reported experiment live in `experiments/` (UniXcoder baseline/weights/focal, CodeBERT, RoBERTa scratch, ModernBERT base/longer/extra-long/augmentation). Execute the relevant script with `bash experiments/<script>.sh`.

Classical baseline: `python train_xgboost.py` trains character n-gram TF-IDF + XGBoost models (creates logs under `logs/`, models under `models/`, and submissions under `predictions/`).

## Evaluation and Submission Generation
- Single model: `python eval.py --model_path <checkpoint_dir> --max_length <len> --batch_size <bs> --do_eval --do_predict --submission_file predictions/<name>.csv`. It reports macro F1 on `test_sample.parquet` and writes predictions for `test.parquet`.
- Batch evaluation: `bash eval.sh` runs the above across all saved checkpoints referenced in the script (ModernBERT variants, UniXcoder variants, CodeBERT, RoBERTa). Ensure the corresponding weights exist under `models/` before running.

## Ensembling
Combine two trained models with `ensemble.py` (weighted averaging, max-prob, or voting). Example:
```bash
bash experiments/run_ensemble_unixcoder_modernbert.sh
```
This script searches weights on `test_sample.parquet`, applies the best config to `test.parquet`, and saves `predictions/ensemble_unixcoder_modernbert.csv`.

## Results
| Exp | Approach | Test F1 | Hybrid F1 | Key Insight |
| --- | --- | --- | --- | --- |
| 1 | XGBoost (10K) | 0.6086 | 0.40 | Char n-grams > word n-grams |
| 2 | XGBoost (100K) | 0.6566 | 0.55 | More data helps |
| 3 | Hyperparameter tuning | 0.6266 | 0.4032 | Identified Hybrid weakness |
| 4 | Transformers (900K) | 0.8598 | 0.7730 | Pretraining crucial |
| 5 | Ensemble (Uni+Code) | 0.8620 | 0.77XX | Marginal gain |
| 6 | Focal Loss | 0.8034 | 0.7629 | **FAILED -5.64%** |
| 7 | ModernBERT (510) | 0.8632 | 0.8228 | Architecture matters |
| 8 | Class Weights | 0.8467 | 0.7904 | **FAILED -1.31%** |
| 9 | Ensemble (Uni+MB) | 0.8642 | 0.8025 | Minimal improvement |
| 10 | Random Cropping | 0.8027 | 0.7571 | **FAILED -6.05%** |
| 11 | **ModernBERT (1024)** | **0.8712** | **0.8242** | **Best completed** |
| 12 | ModernBERT (2048) | 0.8106* | 0.7630* | Promising but incomplete |

Complete experimental timeline. *Exp 12 evaluated at epoch 0.31 (incomplete training due to GPU timeout).*
