"""
Full 900K Training - XGBoost + Character N-grams

Based on 10K experiments: Best config is n=200, d=6, char(3-5)

Tests:
1. char(3-5) + XGB(n=200, d=6, lr=0.1) - best from 10K
2. char(3-6) + XGB(n=200, d=6, lr=0.1) - wider range

Expected: ~20 min per model, Val F1: 0.70-0.75
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from time import time
import pickle
import sys
import os
from datetime import datetime


class Logger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


# Setup logging
log_filename = f'logs/train_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)
logger = Logger(log_filename)
sys.stdout = logger

print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Logs will be saved to: {log_filename}")
print()

# Load Full Data
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)
print("Loading datasets...")
print("  - Loading train.parquet...")
train_df = pd.read_parquet('Task_C/train.parquet')[:50000]
print("  - Loading validation.parquet...")
val_df = pd.read_parquet('Task_C/validation.parquet')[:50000]
print("  - Loading test_sample.parquet...")
test_sample_df = pd.read_parquet('Task_C/test_sample.parquet')
print("  - Loading test.parquet...")
test_df = pd.read_parquet('Task_C/test.parquet')

print("\nExtracting features and labels...")
X_train = train_df['code'].values
y_train = train_df['label'].values

X_val = val_df['code'].values
y_val = val_df['label'].values

X_test = test_sample_df['code'].values
y_test = test_sample_df['label'].values

X_test_final = test_df['code'].values
test_ids = test_df['ID'].values

print("\nDataset sizes:")
print(f"  Train: {len(X_train):,}")
print(f"  Val: {len(X_val):,}")
print(f"  Test (with labels): {len(X_test):,}")
print(f"  Test (final): {len(X_test_final):,}")
print("✓ Data loading completed")

# Experiment 1: char(3-5) + XGB(n=200, d=6)
print("\n" + "="*80)
print("STEP 2: EXPERIMENT 1 - char(3-5) + XGBoost(n=200, d=6, lr=0.1)")
print("="*80)

# Features
print("\n[2.1] Creating TF-IDF features...")
print("  Config: char n-grams (3-5), max_features=10000, min_df=2")
tfidf_35 = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    max_features=10000,
    min_df=2,
    sublinear_tf=True
)

start = time()
print("  - Fitting and transforming training data...")
X_train_35 = tfidf_35.fit_transform(X_train)
print("  - Transforming validation data...")
X_val_35 = tfidf_35.transform(X_val)
print("  - Transforming test data...")
X_test_35 = tfidf_35.transform(X_test)
print("  - Transforming final test data...")
X_test_final_35 = tfidf_35.transform(X_test_final)
print(f"  ✓ Feature extraction completed: {time()-start:.2f}s")
print(f"  Feature matrix shape: {X_train_35.shape}")

# Model
print("\n[2.2] Training XGBoost model...")
print("  Config: n_estimators=200, max_depth=6, lr=0.1")
xgb_35 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

start = time()
print("  - Fitting model (this may take several minutes)...")
xgb_35.fit(X_train_35, y_train)
train_time = time() - start
print(f"  ✓ Training completed: {train_time/60:.2f} min")

# Evaluate
print("\n[2.3] Evaluating model...")
print("  - Predicting on validation set...")
y_val_pred = xgb_35.predict(X_val_35)
print("  - Predicting on test set...")
y_test_pred = xgb_35.predict(X_test_35)

print("  - Calculating F1 scores...")
val_f1 = f1_score(y_val, y_val_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')

print(f"\n[2.4] Results:")
print(f"  Val Macro F1: {val_f1:.4f}")
print(f"  Test Macro F1: {test_f1:.4f}")

print("\nPer-class (Test):")
print(classification_report(y_test, y_test_pred, 
                           target_names=['Human', 'AI', 'Hybrid', 'Adversarial'],
                           digits=4))

# Save
print("\n[2.5] Saving models...")
print("  - Saving XGBoost model...")
with open('models/xgb_char35_900k.pkl', 'wb') as f:
    pickle.dump(xgb_35, f)
print("  - Saving TF-IDF vectorizer...")
with open('models/tfidf_char35_900k.pkl', 'wb') as f:
    pickle.dump(tfidf_35, f)
print("  ✓ Models saved")

# Predictions on final test
print("\n[2.6] Generating predictions on final test set...")
y_test_final_35 = xgb_35.predict(X_test_final_35)
exp1_results = {'val_f1': val_f1, 'test_f1': test_f1, 'predictions': y_test_final_35}
print("  ✓ Experiment 1 completed")

# Experiment 2: char(3-6) + XGB(n=200, d=6)
print("\n" + "="*80)
print("STEP 3: EXPERIMENT 2 - char(3-6) + XGBoost(n=200, d=6, lr=0.1)")
print("="*80)

# Features
print("\n[3.1] Creating TF-IDF features...")
print("  Config: char n-grams (3-6), max_features=10000, min_df=2")
tfidf_36 = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 6),
    max_features=10000,
    min_df=2,
    sublinear_tf=True
)

start = time()
print("  - Fitting and transforming training data...")
X_train_36 = tfidf_36.fit_transform(X_train)
print("  - Transforming validation data...")
X_val_36 = tfidf_36.transform(X_val)
print("  - Transforming test data...")
X_test_36 = tfidf_36.transform(X_test)
print("  - Transforming final test data...")
X_test_final_36 = tfidf_36.transform(X_test_final)
print(f"  ✓ Feature extraction completed: {time()-start:.2f}s")
print(f"  Feature matrix shape: {X_train_36.shape}")

# Model
print("\n[3.2] Training XGBoost model...")
print("  Config: n_estimators=200, max_depth=6, lr=0.1")
xgb_36 = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

start = time()
print("  - Fitting model (this may take several minutes)...")
xgb_36.fit(X_train_36, y_train)
train_time = time() - start
print(f"  ✓ Training completed: {train_time/60:.2f} min")

# Evaluate
print("\n[3.3] Evaluating model...")
print("  - Predicting on validation set...")
y_val_pred = xgb_36.predict(X_val_36)
print("  - Predicting on test set...")
y_test_pred = xgb_36.predict(X_test_36)

print("  - Calculating F1 scores...")
val_f1 = f1_score(y_val, y_val_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')

print(f"\n[3.4] Results:")
print(f"  Val Macro F1: {val_f1:.4f}")
print(f"  Test Macro F1: {test_f1:.4f}")

print("\nPer-class (Test):")
print(classification_report(y_test, y_test_pred, 
                           target_names=['Human', 'AI', 'Hybrid', 'Adversarial'],
                           digits=4))

# Save
print("\n[3.5] Saving models...")
print("  - Saving XGBoost model...")
with open('models/xgb_char36_900k.pkl', 'wb') as f:
    pickle.dump(xgb_36, f)
print("  - Saving TF-IDF vectorizer...")
with open('models/tfidf_char36_900k.pkl', 'wb') as f:
    pickle.dump(tfidf_36, f)
print("  ✓ Models saved")

# Predictions on final test
print("\n[3.6] Generating predictions on final test set...")
y_test_final_36 = xgb_36.predict(X_test_final_36)
exp2_results = {'val_f1': val_f1, 'test_f1': test_f1, 'predictions': y_test_final_36}
print("  ✓ Experiment 2 completed")

# Compare & Generate Submission
print("\n" + "="*80)
print("STEP 4: FINAL COMPARISON & SUBMISSION GENERATION")
print("="*80)

print("\n[4.1] Comparing experiment results...")
results_df = pd.DataFrame([
    {'Experiment': 'char(3-5)', 'Val F1': exp1_results['val_f1'], 'Test F1': exp1_results['test_f1']},
    {'Experiment': 'char(3-6)', 'Val F1': exp2_results['val_f1'], 'Test F1': exp2_results['test_f1']}
])

print(results_df.to_string(index=False))

# Choose best
print("\n[4.2] Selecting best model...")
best_idx = results_df['Val F1'].idxmax()
best_exp = results_df.loc[best_idx, 'Experiment']
print(f"  ✓ Best model: {best_exp}")

# Save both model predictions
print("\n[4.3] Preparing submission files...")
submission_35 = pd.DataFrame({
    'ID': test_ids,
    'label': exp1_results['predictions']
})

submission_36 = pd.DataFrame({
    'ID': test_ids,
    'label': exp2_results['predictions']
})

# Generate submission with best model
best_predictions = exp1_results['predictions'] if best_idx == 0 else exp2_results['predictions']

submission = pd.DataFrame({
    'ID': test_ids,
    'label': best_predictions
})

# Save all submissions
print("  - Saving char(3-5) submission...")
submission_35.to_csv('predictions/submission_char35_900k.csv', index=False)
print("  - Saving char(3-6) submission...")
submission_36.to_csv('predictions/submission_char36_900k.csv', index=False)
print("  - Saving best model submission...")
submission.to_csv('predictions/submission_900k.csv', index=False)

print(f"\n[4.4] Submissions saved:")
print(f"  char(3-5): predictions/submission_char35_900k.csv")
print(f"  char(3-6): predictions/submission_char36_900k.csv")
print(f"  Best ({best_exp}): predictions/submission_900k.csv")
print(f"\n[4.5] Best submission preview:")
print(submission.head())

print("\n" + "="*80)
print("ALL STEPS COMPLETED SUCCESSFULLY")
print("="*80)
print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Close logger
sys.stdout = logger.terminal
logger.close()
print(f"Logs saved to: {log_filename}")
