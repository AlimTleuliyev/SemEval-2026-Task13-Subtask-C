"""
Ensemble predictions from multiple models
Supports: UniXcoder, CodeBERT, ModernBERT-base, ModernBERT-large
Combines probability outputs from multiple models
Automatically finds best ensemble configuration on test_sample
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


class CodeDataset(Dataset):
    """Simple dataset for inference"""
    def __init__(self, codes, tokenizer, max_length=512):
        self.codes = codes
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
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def get_predictions(model_path, tokenizer_path, test_df, max_length=510, batch_size=32, device='cuda', model_type='roberta'):
    """
    Get probability predictions from a single model

    Args:
        model_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        test_df: DataFrame with 'code' column
        max_length: Max sequence length
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'
        model_type: 'roberta' (UniXcoder/CodeBERT) or 'modernbert'
    """
    print(f"\nLoading model from: {model_path}")
    print(f"Model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load appropriate model class based on type
    if model_type == 'modernbert':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("Using AutoModelForSequenceClassification (ModernBERT)")
    else:
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        print("Using RobertaForSequenceClassification (UniXcoder/CodeBERT)")

    model.to(device)
    model.eval()

    dataset = CodeDataset(test_df['code'].values, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def ensemble_predictions(probs_list, weights=None, method='weighted_avg'):
    """
    Ensemble multiple model predictions

    Args:
        probs_list: List of probability arrays from different models
        weights: List of weights for each model (must sum to 1)
        method: 'weighted_avg', 'max_prob', or 'voting'
    """
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)

    if method == 'weighted_avg':
        # Weighted average of probabilities
        ensemble_probs = sum(w * p for w, p in zip(weights, probs_list))
        predictions = np.argmax(ensemble_probs, axis=1)

    elif method == 'max_prob':
        # Take prediction with highest confidence
        max_probs = [np.max(p, axis=1) for p in probs_list]
        best_model_idx = np.argmax(max_probs, axis=0)
        predictions = np.array([probs_list[idx][i].argmax()
                               for i, idx in enumerate(best_model_idx)])
        ensemble_probs = None

    elif method == 'voting':
        # Majority voting
        preds_list = [np.argmax(p, axis=1) for p in probs_list]
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=np.vstack(preds_list)
        )
        ensemble_probs = None

    return predictions, ensemble_probs


def evaluate_predictions(predictions, labels):
    """Compute F1 macro score"""
    f1_macro = f1_score(labels, predictions, average='macro')
    return f1_macro


def find_best_ensemble(probs_list, labels, model_names=['Model1', 'Model2']):
    """
    Try all ensemble methods and weight combinations to find best configuration

    Args:
        probs_list: List of probability arrays from different models
        labels: Ground truth labels
        model_names: Names of models for reporting

    Returns:
        best_config: Dictionary with best configuration and results
    """
    print("\n" + "="*80)
    print("TESTING ALL ENSEMBLE CONFIGURATIONS")
    print("="*80)

    # First show individual model performance as baseline
    print("\nINDIVIDUAL MODEL PERFORMANCE (Baseline):")
    print("-" * 80)
    for i, (probs, name) in enumerate(zip(probs_list, model_names)):
        preds = np.argmax(probs, axis=1)
        f1 = evaluate_predictions(preds, labels)
        print(f"{name:15} | F1: {f1:.4f}")

    print("\n" + "-" * 80)
    print("ENSEMBLE COMBINATIONS:")
    print("-" * 80)

    results = []

    # Test methods
    methods = ['weighted_avg', 'max_prob', 'voting']

    # Test weight combinations for weighted_avg
    # Try all combinations from 0.1 to 0.9 for model1 (model2 = 1 - model1)
    weight_combinations = [
        [0.1, 0.9],   # Heavily favor Model 2
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],   # Equal
        [0.6, 0.4],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.9, 0.1],   # Heavily favor Model 1
    ]

    for method in methods:
        if method == 'weighted_avg':
            # Test different weight combinations
            for weights in weight_combinations:
                preds, _ = ensemble_predictions(probs_list, weights=weights, method=method)
                f1 = evaluate_predictions(preds, labels)

                config = {
                    'method': method,
                    'weights': weights,
                    'weight_str': f"{model_names[0]}={weights[0]:.1f}, {model_names[1]}={weights[1]:.1f}",
                    'f1_macro': f1,
                    'predictions': preds
                }
                results.append(config)

                print(f"{method:15} | {config['weight_str']:25} | F1: {f1:.4f}")

        else:
            # For voting and max_prob, weights don't matter
            preds, _ = ensemble_predictions(probs_list, weights=None, method=method)
            f1 = evaluate_predictions(preds, labels)

            config = {
                'method': method,
                'weights': None,
                'weight_str': 'N/A',
                'f1_macro': f1,
                'predictions': preds
            }
            results.append(config)

            print(f"{method:15} | {'N/A':25} | F1: {f1:.4f}")

    # Find best configuration
    best_config = max(results, key=lambda x: x['f1_macro'])

    print("\n" + "="*80)
    print("BEST CONFIGURATION FOUND")
    print("="*80)
    print(f"Method: {best_config['method']}")
    print(f"Weights: {best_config['weight_str']}")
    print(f"F1 Macro: {best_config['f1_macro']:.4f}")

    return best_config, results


def main():
    parser = argparse.ArgumentParser(description='Ensemble multiple transformer models')

    # Model 1 configuration
    parser.add_argument('--model1_path', type=str,
                       default='./models/unixcoder_full/final',
                       help='Path to first model')
    parser.add_argument('--model1_tokenizer', type=str,
                       default='microsoft/unixcoder-base',
                       help='Tokenizer for first model')
    parser.add_argument('--model1_type', type=str,
                       default='roberta',
                       choices=['roberta', 'modernbert'],
                       help='Type of first model')
    parser.add_argument('--model1_name', type=str,
                       default='Model1',
                       help='Display name for first model')

    # Model 2 configuration
    parser.add_argument('--model2_path', type=str,
                       default='./models/codebert_full/final',
                       help='Path to second model')
    parser.add_argument('--model2_tokenizer', type=str,
                       default='microsoft/codebert-base',
                       help='Tokenizer for second model')
    parser.add_argument('--model2_type', type=str,
                       default='roberta',
                       choices=['roberta', 'modernbert'],
                       help='Type of second model')
    parser.add_argument('--model2_name', type=str,
                       default='Model2',
                       help='Display name for second model')

    # Data
    parser.add_argument('--test_sample_file', type=str,
                       default='Task_C/test_sample.parquet',
                       help='Test sample file (with labels for finding best config)')
    parser.add_argument('--test_file', type=str,
                       default='Task_C/test.parquet',
                       help='Test file (for final predictions)')
    parser.add_argument('--max_length', type=int, default=510,
                       help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')

    # Output
    parser.add_argument('--output_file', type=str,
                       default='./predictions/ensemble_submission.csv',
                       help='Output submission file')
    parser.add_argument('--skip_test_optimization', action='store_true',
                       help='Skip finding best config on test_sample, use defaults')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test_sample data (with labels) to find best ensemble config
    print(f"\nLoading test_sample data from: {args.test_sample_file}")
    test_sample_df = pd.read_parquet(args.test_sample_file)
    print(f"Test sample: {len(test_sample_df)} samples")

    # Get predictions on test_sample from each model
    print("\n" + "="*80)
    print("STEP 1: GET PREDICTIONS ON TEST_SAMPLE (with labels)")
    print("="*80)

    print(f"\nMODEL 1: {args.model1_name}")
    print("-" * 80)
    model1_probs_sample = get_predictions(
        args.model1_path,
        args.model1_tokenizer,
        test_sample_df,
        args.max_length,
        args.batch_size,
        device,
        model_type=args.model1_type
    )
    print(f"{args.model1_name} predictions shape: {model1_probs_sample.shape}")

    print(f"\nMODEL 2: {args.model2_name}")
    print("-" * 80)
    model2_probs_sample = get_predictions(
        args.model2_path,
        args.model2_tokenizer,
        test_sample_df,
        args.max_length,
        args.batch_size,
        device,
        model_type=args.model2_type
    )
    print(f"{args.model2_name} predictions shape: {model2_probs_sample.shape}")

    # Find best ensemble configuration
    probs_list_sample = [model1_probs_sample, model2_probs_sample]
    labels = test_sample_df['label'].values

    print("\n" + "="*80)
    print("STEP 2: FIND BEST ENSEMBLE CONFIGURATION")
    print("="*80)

    best_config, all_results = find_best_ensemble(
        probs_list_sample,
        labels,
        model_names=[args.model1_name, args.model2_name]
    )

    # Print detailed results for best config
    print("\n" + "="*80)
    print("DETAILED RESULTS FOR BEST CONFIGURATION")
    print("="*80)
    print(f"\nMethod: {best_config['method']}")
    print(f"Weights: {best_config['weight_str']}")
    print(f"Macro F1: {best_config['f1_macro']:.4f}")

    print("\nPer-class performance:")
    print(classification_report(
        labels,
        best_config['predictions'],
        target_names=['Human', 'AI', 'Hybrid', 'Adversarial'],
        digits=4
    ))

    # Now get predictions on actual test.parquet using best config
    print("\n" + "="*80)
    print("STEP 3: GENERATE FINAL PREDICTIONS ON TEST.PARQUET")
    print("="*80)

    print(f"\nLoading test data from: {args.test_file}")
    test_df = pd.read_parquet(args.test_file)
    print(f"Test samples: {len(test_df)}")

    print(f"\nMODEL 1: {args.model1_name}")
    print("-" * 80)
    model1_probs = get_predictions(
        args.model1_path,
        args.model1_tokenizer,
        test_df,
        args.max_length,
        args.batch_size,
        device,
        model_type=args.model1_type
    )

    print(f"\nMODEL 2: {args.model2_name}")
    print("-" * 80)
    model2_probs = get_predictions(
        args.model2_path,
        args.model2_tokenizer,
        test_df,
        args.max_length,
        args.batch_size,
        device,
        model_type=args.model2_type
    )

    # Use best configuration
    probs_list = [model1_probs, model2_probs]
    final_predictions, ensemble_probs = ensemble_predictions(
        probs_list,
        weights=best_config['weights'],
        method=best_config['method']
    )

    # Create submission
    submission_df = pd.DataFrame({
        'ID': test_df['ID'].values,
        'label': final_predictions
    })

    # Save
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    submission_df.to_csv(args.output_file, index=False)

    print("\n" + "="*80)
    print("✅ ENSEMBLE COMPLETE!")
    print("="*80)
    print(f"\nBest configuration:")
    print(f"  Method: {best_config['method']}")
    print(f"  Weights: {best_config['weight_str']}")
    print(f"  Test F1 (on test_sample): {best_config['f1_macro']:.4f}")
    print(f"\nSaved submission to: {args.output_file}")
    print(f"\nPrediction distribution:")
    print(submission_df['label'].value_counts().sort_index())

    # Also save probabilities for analysis
    if ensemble_probs is not None:
        prob_file = args.output_file.replace('.csv', '_probs.npy')
        np.save(prob_file, ensemble_probs)
        print(f"\nSaved probabilities to: {prob_file}")

    # Save all ensemble results
    results_file = args.output_file.replace('.csv', '_all_results.txt')
    with open(results_file, 'w') as f:
        f.write("ALL ENSEMBLE CONFIGURATIONS TESTED\n")
        f.write("="*80 + "\n\n")
        for i, result in enumerate(all_results, 1):
            f.write(f"{i}. Method: {result['method']:15} | Weights: {result['weight_str']:25} | F1: {result['f1_macro']:.4f}\n")
        f.write("\n" + "="*80 + "\n")
        f.write(f"BEST: {best_config['method']:15} | {best_config['weight_str']:25} | F1: {best_config['f1_macro']:.4f}\n")
    print(f"Saved all results to: {results_file}")


if __name__ == '__main__':
    main()
