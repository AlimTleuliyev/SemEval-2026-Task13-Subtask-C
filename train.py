"""
Train transformer models for AI-generated code detection
Simple script - no over-engineering, uses Trainer's built-in logging
"""
import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from sklearn.metrics import f1_score, classification_report


class GPUUsageCallback(TrainerCallback):
    """Callback to log GPU memory usage during training"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and logs is not None:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logs['gpu_allocated_gb'] = round(allocated, 2)
            logs['gpu_reserved_gb'] = round(reserved, 2)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses training on hard examples

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights (same as in weighted CE)
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        """
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedTrainer(Trainer):
    """Trainer with weighted loss (cross-entropy or focal) for handling class imbalance"""
    def __init__(self, class_weights=None, use_focal_loss=False, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss

        if use_focal_loss:
            self.loss_fct = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        elif class_weights is not None:
            self.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class CodeDataset(Dataset):
    """
    Dataset for code classification with optional random cropping

    Random cropping augmentation:
    - Instead of always taking first N tokens, randomly crops different windows
    - Helps model see different parts of long code files
    - Disabled for validation/test (use_random_crop=False)
    """
    def __init__(self, codes, labels, tokenizer, max_length=512, use_random_crop=False):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_random_crop = use_random_crop

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = str(self.codes[idx])

        # If random cropping enabled, tokenize first to check length
        if self.use_random_crop:
            # Tokenize without truncation to get full length
            full_encoding = self.tokenizer(
                code,
                truncation=False,
                return_tensors='pt'
            )
            input_ids = full_encoding['input_ids'].flatten()
            full_length = len(input_ids)

            # If longer than max_length, crop randomly
            if full_length > self.max_length:
                # Random start position
                max_start = full_length - self.max_length
                start_idx = random.randint(0, max_start)
                end_idx = start_idx + self.max_length

                input_ids = input_ids[start_idx:end_idx]
                attention_mask = torch.ones_like(input_ids)
            else:
                # Shorter than max_length, pad normally
                encoding = self.tokenizer(
                    code,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].flatten()
                attention_mask = encoding['attention_mask'].flatten()
        else:
            # Standard truncation/padding (no random cropping)
            encoding = self.tokenizer(
                code,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()

        # Ensure valid label (0-3 for 4 classes)
        label = int(self.labels[idx])
        if label < 0 or label > 3:
            raise ValueError(f"Invalid label {label} at index {idx}. Expected 0-3.")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute macro F1 for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1_macro = f1_score(labels, predictions, average='macro')
    return {'f1_macro': f1_macro}


def load_data(args):
    """Load train, validation, and test data"""
    print(f"\nLoading data...")
    
    # Load train data
    train_df = pd.read_parquet('Task_C/train.parquet')
    if args.train_size < len(train_df):
        train_df = train_df.sample(n=args.train_size, random_state=42)
    print(f"Train: {len(train_df)} samples")
    
    # Load validation data
    val_df = pd.read_parquet('Task_C/validation.parquet')
    if args.val_size < len(val_df):
        val_df = val_df.sample(n=args.val_size, random_state=42)
    print(f"Validation: {len(val_df)} samples")
    
    # Load test data (with labels for evaluation)
    test_sample_df = pd.read_parquet('Task_C/test_sample.parquet')
    print(f"Test (with labels): {len(test_sample_df)} samples")
    
    # Load test data (no labels, for prediction)
    test_df = pd.read_parquet('Task_C/test.parquet')
    print(f"Test (for prediction): {len(test_df)} samples")
    
    return train_df, val_df, test_sample_df, test_df


def create_model(args):
    """Create model and tokenizer based on model_name"""
    print(f"\nLoading model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.model_name == 'roberta-scratch':
        # Train from scratch
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            num_labels=4,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12
        )
        model = RobertaForSequenceClassification(config)
        print("Initialized RoBERTa from scratch")
    elif args.model_name == 'modernbert-base':
        # ModernBERT uses AutoModelForSequenceClassification
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=4,
            trust_remote_code=True  # ModernBERT may need this
        )
        print(f"Loaded ModernBERT from {args.model_path}")
    else:
        # Load pretrained (CodeBERT, UniXcoder, etc.)
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_path,
            num_labels=4
        )
        print(f"Loaded pretrained model from {args.model_path}")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train transformer models for code classification')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['codebert', 'unixcoder', 'roberta-scratch', 'modernbert-base'],
                        help='Model to train')
    parser.add_argument('--model_path', type=str, required=True,
                        help='HuggingFace model path')
    
    # Data arguments
    parser.add_argument('--train_size', type=int, default=900000,
                        help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=200000,
                        help='Number of validation samples')
    parser.add_argument('--max_length', type=int, default=510,
                        help='Max sequence length')
    parser.add_argument('--use_random_crop', action='store_true',
                        help='Use random cropping for data augmentation (training only)')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per-device batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Logging/Saving
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Logging frequency')
    parser.add_argument('--save_strategy', type=str, default='steps',
                        choices=['steps', 'epoch'],
                        help='Save strategy')
    parser.add_argument('--save_steps', type=int, default=5000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=5000,
                        help='Eval on validation every N steps')
    
    # Execution flags
    parser.add_argument('--do_train', action='store_true',
                        help='Run training')
    parser.add_argument('--do_eval', action='store_true',
                        help='Evaluate on test_sample.parquet after training')
    parser.add_argument('--do_predict', action='store_true',
                        help='Generate predictions on test.parquet')
    parser.add_argument('--submission_file', type=str, default='submission.csv',
                        help='Submission file path')
    parser.add_argument('--class_weights', type=str, default=None,
                        help='Class weights as comma-separated values: "w0,w1,w2,w3"')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal Loss instead of weighted CE')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focusing parameter for Focal Loss (default: 2.0)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., ./models/unixcoder_focal/checkpoint-1000)')

    args = parser.parse_args()
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_df, val_df, test_sample_df, test_df = load_data(args)
    
    # Create model
    model, tokenizer = create_model(args)
    
    # Create datasets
    print(f"\nCreating datasets (max_length={args.max_length})...")
    if args.use_random_crop:
        print("✅ Random cropping ENABLED for training (data augmentation)")

    train_dataset = CodeDataset(
        train_df['code'].values,
        train_df['label'].values,
        tokenizer,
        max_length=args.max_length,
        use_random_crop=args.use_random_crop  # Enable for training
    )

    val_dataset = CodeDataset(
        val_df['code'].values,
        val_df['label'].values,
        tokenizer,
        max_length=args.max_length,
        use_random_crop=False  # Always disabled for validation
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        report_to='none',
        disable_tqdm=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),  # Mixed precision training on CUDA
        gradient_checkpointing=False,  # Disabled for speed (uses more memory but faster)
        optim='adamw_torch_fused' if torch.cuda.is_available() else 'adamw_torch',  # Fused optimizer
    )
    
    # Parse class weights if provided
    class_weights_tensor = None
    if args.class_weights:
        weights = [float(w) for w in args.class_weights.split(',')]
        class_weights_tensor = torch.tensor(weights, dtype=torch.float)

        # Move to correct device
        if torch.cuda.is_available():
            class_weights_tensor = class_weights_tensor.cuda()
        elif torch.backends.mps.is_available():
            class_weights_tensor = class_weights_tensor.to('mps')

        print(f"\nUsing class weights: {weights}")
        print(f"Class weights tensor device: {class_weights_tensor.device}")

    # Trainer
    callbacks = []
    if torch.cuda.is_available():
        callbacks.append(GPUUsageCallback())

    # Use WeightedTrainer if class weights are provided, otherwise use standard Trainer
    if class_weights_tensor is not None:
        trainer = WeightedTrainer(
            class_weights=class_weights_tensor,
            use_focal_loss=args.use_focal_loss,
            focal_gamma=args.focal_gamma,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        if args.use_focal_loss:
            print(f"Using Focal Loss with gamma={args.focal_gamma}")
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
    
    # Train
    if args.do_train:
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
        if args.resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Save final model
        print(f"\nSaving final model to {args.output_dir}/final")
        model.save_pretrained(f'{args.output_dir}/final')
        tokenizer.save_pretrained(f'{args.output_dir}/final')
    
    # Evaluate on test_sample.parquet
    if args.do_eval:
        print("\n" + "="*80)
        print("EVALUATION ON TEST SET (with labels)")
        print("="*80)
        
        test_eval_dataset = CodeDataset(
            test_sample_df['code'].values,
            test_sample_df['label'].values,
            tokenizer,
            max_length=args.max_length,
            use_random_crop=False  # Always disabled for testing
        )
        
        results = trainer.evaluate(test_eval_dataset)
        print(f"\nTest F1 Macro: {results['eval_f1_macro']:.4f}")
        
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
        
        # Create dataset without labels (use dummy labels)
        test_pred_dataset = CodeDataset(
            test_df['code'].values,
            np.zeros(len(test_df), dtype=int),  # Dummy labels
            tokenizer,
            max_length=args.max_length,
            use_random_crop=False  # Always disabled for prediction
        )
        
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
