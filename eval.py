"""
Evaluate a saved transformer model on test set
Usage: python eval_transformer.py --model_path ./models/codebert_full/final --submission_file ./predictions/eval_submission.csv
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    ModernBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import f1_score, classification_report


class CodeDataset(Dataset):
    """Dataset for code classification"""
    def __init__(self, codes, labels, tokenizer, max_length=510):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.codes[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label = int(self.labels[idx])
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute macro F1 for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1_macro = f1_score(labels, predictions, average='macro')
    return {'f1_macro': f1_macro}


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved transformer model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model (e.g., ./models/codebert_full/final)')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to tokenizer (default: same as model_path, or infer from model name)')
    parser.add_argument('--max_length', type=int, default=510,
                        help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--do_eval', action='store_true',
                        help='Evaluate on test_sample.parquet (with labels)')
    parser.add_argument('--do_predict', action='store_true',
                        help='Generate predictions on test.parquet')
    parser.add_argument('--submission_file', type=str, default='predictions/eval_submission.csv',
                        help='Submission file path')
    
    args = parser.parse_args()
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Determine tokenizer path and model type
    tokenizer_path = args.tokenizer_path
    is_modernbert = 'modernbert' in args.model_path.lower()

    if tokenizer_path is None:
        # First try to load from model_path (works for final/ directories)
        if os.path.exists(os.path.join(args.model_path, 'tokenizer_config.json')):
            tokenizer_path = args.model_path
        else:
            # For checkpoints, infer original model from path
            print("⚠️  Checkpoint doesn't contain tokenizer files")
            if 'modernbert' in args.model_path.lower():
                tokenizer_path = 'answerdotai/ModernBERT-base'
                print(f"   Using tokenizer from: {tokenizer_path}")
            elif 'codebert' in args.model_path.lower():
                tokenizer_path = 'microsoft/codebert-base'
                print(f"   Using tokenizer from: {tokenizer_path}")
            elif 'unixcoder' in args.model_path.lower():
                tokenizer_path = 'microsoft/unixcoder-base'
                print(f"   Using tokenizer from: {tokenizer_path}")
            elif 'roberta' in args.model_path.lower():
                # For RoBERTa scratch, try to find final/ directory
                base_dir = args.model_path.split('/checkpoint')[0]
                final_dir = os.path.join(base_dir, 'final')
                if os.path.exists(final_dir):
                    tokenizer_path = final_dir
                    print(f"   Using tokenizer from: {tokenizer_path}")
                else:
                    tokenizer_path = 'roberta-base'
                    print(f"   Using tokenizer from: {tokenizer_path}")
            else:
                tokenizer_path = args.model_path
                print(f"   Attempting to load tokenizer from: {tokenizer_path}")

    # Load model and tokenizer
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"Loading model from: {args.model_path}")
    if is_modernbert:
        print("  Model type: ModernBERT (AutoModelForSequenceClassification)")
        model = ModernBertForSequenceClassification.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )
    else:
        print("  Model type: RoBERTa-based (RobertaForSequenceClassification)")
        model = RobertaForSequenceClassification.from_pretrained(args.model_path)

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training arguments (for Trainer, even though we're not training)
    training_args = TrainingArguments(
        output_dir='./eval_output',
        per_device_eval_batch_size=args.batch_size,
        report_to='none',
        disable_tqdm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )
    
    # Evaluate on test_sample.parquet
    if args.do_eval:
        print("\n" + "="*80)
        print("EVALUATION ON TEST SET (with labels)")
        print("="*80)
        
        print("Loading test_sample.parquet...")
        test_sample_df = pd.read_parquet('Task_C/test_sample.parquet')
        print(f"Test samples: {len(test_sample_df)}")
        
        test_eval_dataset = CodeDataset(
            test_sample_df['code'].values,
            test_sample_df['label'].values,
            tokenizer,
            max_length=args.max_length
        )
        
        print("\nEvaluating...")
        results = trainer.evaluate(test_eval_dataset)
        print(f"\nTest F1 Macro: {results['eval_f1_macro']:.4f}")
        print(f"Test Loss: {results['eval_loss']:.4f}")
        
        # Get predictions for detailed report
        predictions = trainer.predict(test_eval_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        print("\nPer-class performance:")
        print(classification_report(
            labels, preds,
            target_names=['Human', 'AI', 'Hybrid', 'Adversarial'],
            digits=4
        ))
    
    # Generate predictions on test.parquet
    if args.do_predict:
        print("\n" + "="*80)
        print("GENERATING PREDICTIONS FOR SUBMISSION")
        print("="*80)
        
        print("Loading test.parquet...")
        test_df = pd.read_parquet('Task_C/test.parquet')
        print(f"Test samples: {len(test_df)}")
        
        # Create dataset without labels (use dummy labels)
        test_pred_dataset = CodeDataset(
            test_df['code'].values,
            np.zeros(len(test_df), dtype=int),  # Dummy labels
            tokenizer,
            max_length=args.max_length
        )
        
        print("\nGenerating predictions...")
        predictions = trainer.predict(test_pred_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'ID': test_df['ID'].values,
            'label': preds
        })
        
        # Save submission file
        os.makedirs(os.path.dirname(args.submission_file), exist_ok=True)
        submission_df.to_csv(args.submission_file, index=False)
        print(f"\nSaved predictions to: {args.submission_file}")
        print(f"Predictions shape: {submission_df.shape}")
        print(f"\nPrediction distribution:")
        print(submission_df['label'].value_counts().sort_index())
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
