#!/usr/bin/env python3
"""
Generate publication-quality plots for NLP701 Assignment 2 Report
Focuses on key findings: ModernBERT architecture + longer sequences
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Create figures directory
output_dir = Path("report_figures")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("GENERATING REPORT FIGURES")
print("=" * 80)
print()

# ============================================================================
# FIGURE 1: Learning Curves - Main Models
# ============================================================================
print("📊 Figure 1: Learning Curves (Main Models)")

fig, ax = plt.subplots(figsize=(8, 5))

# Load training histories for key models
models_to_plot = [
    {
        'path': 'models/unixcoder_full/checkpoint-24612/trainer_state.json',
        'name': 'UniXcoder (510 tokens)',
        'color': '#2ecc71',
        'linestyle': '-'
    },
    {
        'path': 'models/modernbert_base_full/checkpoint-10548/trainer_state.json',
        'name': 'ModernBERT (510 tokens)',
        'color': '#3498db',
        'linestyle': '-'
    },
    {
        'path': 'models/modernbert_longer_full/checkpoint-15625/trainer_state.json',
        'name': 'ModernBERT (1024 tokens)',
        'color': '#e74c3c',
        'linestyle': '-',
        'linewidth': 2.5
    }
]

for model_info in models_to_plot:
    try:
        with open(model_info['path'], 'r') as f:
            state = json.load(f)

        # Extract validation F1 scores
        log_history = state['log_history']
        eval_entries = [entry for entry in log_history if 'eval_f1_macro' in entry]

        if eval_entries:
            steps = [entry['step'] for entry in eval_entries]
            f1_scores = [entry['eval_f1_macro'] for entry in eval_entries]

            linewidth = model_info.get('linewidth', 2)
            ax.plot(steps, f1_scores,
                   label=model_info['name'],
                   color=model_info['color'],
                   linestyle=model_info['linestyle'],
                   linewidth=linewidth,
                   marker='o',
                   markersize=4,
                   markevery=max(1, len(steps)//10))

            print(f"  ✓ {model_info['name']}: {len(steps)} evaluation points, best F1 = {max(f1_scores):.4f}")
    except Exception as e:
        print(f"  ✗ {model_info['name']}: {e}")

ax.set_xlabel('Training Steps')
ax.set_ylabel('Validation F1 Score (Macro)')
ax.set_title('Learning Curves: Architecture and Sequence Length Impact')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.75, 0.88])

plt.tight_layout()
plt.savefig(output_dir / 'fig1_learning_curves.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig1_learning_curves.pdf', bbox_inches='tight')
print(f"  💾 Saved: {output_dir}/fig1_learning_curves.png")
plt.close()
print()

# ============================================================================
# FIGURE 2: Sequence Length Impact (with Hybrid Class Focus)
# ============================================================================
print("📊 Figure 2: Sequence Length Impact (Hybrid Class Breakthrough)")

# Data from findings.md
sequence_data = {
    '510 tokens\n(60% coverage)': {
        'overall': 0.8632,
        'hybrid': 0.8228,
        'adversarial': 0.7937,
        'ai': 0.8578,
        'human': 0.9787
    },
    '1024 tokens\n(84% coverage)': {
        'overall': 0.8712,
        'hybrid': 0.8242,
        'adversarial': 0.8281,
        'ai': 0.8565,
        'human': 0.9759
    },
    '2048 tokens\n(95% coverage)': {
        'overall': 0.88,  # Projected
        'hybrid': 0.84,   # Projected
        'adversarial': 0.84,  # Projected
        'ai': 0.86,       # Projected
        'human': 0.975    # Projected
    }
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left plot: Overall F1 and Hybrid F1
x_pos = np.arange(len(sequence_data))
width = 0.35

overall_scores = [data['overall'] for data in sequence_data.values()]
hybrid_scores = [data['hybrid'] for data in sequence_data.values()]

bars1 = ax1.bar(x_pos - width/2, overall_scores, width,
               label='Overall F1 (Macro)', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, hybrid_scores, width,
               label='Hybrid Class F1', color='#e74c3c', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Sequence Length Configuration')
ax1.set_ylabel('F1 Score')
ax1.set_title('Impact of Sequence Length on Performance')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(sequence_data.keys())
ax1.legend(loc='lower right')
ax1.set_ylim([0.75, 0.92])
ax1.grid(True, alpha=0.3, axis='y')

# Add annotation for 2048 projection
ax1.text(2, 0.76, '* Projected', fontsize=7, style='italic', ha='center')

# Right plot: Per-class breakdown for best model (1024 tokens)
classes = ['Human', 'AI-Gen', 'Hybrid', 'Adversarial']
best_model_scores = [0.9759, 0.8565, 0.8242, 0.8281]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

bars = ax2.bar(classes, best_model_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=9)

ax2.set_ylabel('F1 Score')
ax2.set_title('Per-Class Performance: ModernBERT (1024 tokens)')
ax2.set_ylim([0.75, 1.0])
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0.8712, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Overall Macro F1')
ax2.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'fig2_sequence_length_impact.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig2_sequence_length_impact.pdf', bbox_inches='tight')
print(f"  💾 Saved: {output_dir}/fig2_sequence_length_impact.png")
plt.close()
print()

# ============================================================================
# FIGURE 3: Model Comparison (All Major Experiments)
# ============================================================================
print("📊 Figure 3: Model Comparison")

# Data from evaluation results
models_data = {
    'XGBoost\n(Baseline)': {'f1': 0.72, 'type': 'classical'},
    'RoBERTa\n(Scratch)': {'f1': 0.7388, 'type': 'failed'},
    'CodeBERT': {'f1': 0.8446, 'type': 'baseline'},
    'UniXcoder': {'f1': 0.8598, 'type': 'baseline'},
    'Focal Loss': {'f1': 0.8034, 'type': 'failed'},
    'Class Weights': {'f1': 0.8467, 'type': 'failed'},
    'Random\nCropping': {'f1': 0.8027, 'type': 'failed'},
    'ModernBERT\n(510)': {'f1': 0.8632, 'type': 'modern'},
    'Ensemble\n(Uni+MB)': {'f1': 0.8642, 'type': 'modern'},
    'ModernBERT\n(1024)': {'f1': 0.8712, 'type': 'best'}
}

fig, ax = plt.subplots(figsize=(12, 6))

# Color code by type
color_map = {
    'classical': '#95a5a6',
    'failed': '#e67e22',
    'baseline': '#3498db',
    'modern': '#9b59b6',
    'best': '#27ae60'
}

models = list(models_data.keys())
f1_scores = [data['f1'] for data in models_data.values()]
colors = [color_map[data['type']] for data in models_data.values()]

bars = ax.barh(models, f1_scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    width = bar.get_width()
    ax.text(width + 0.003, bar.get_y() + bar.get_height()/2,
            f'{score:.4f}',
            ha='left', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Test F1 Score (Macro)')
ax.set_title('Model Performance Comparison: From Baselines to Best Model')
ax.set_xlim([0.70, 0.88])
ax.grid(True, alpha=0.3, axis='x')

# Add vertical line for target
ax.axvline(x=0.87, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.87)')

# Create legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_map['classical'], label='Classical ML'),
    Patch(facecolor=color_map['baseline'], label='Transformer Baselines'),
    Patch(facecolor=color_map['failed'], label='Failed Approaches'),
    Patch(facecolor=color_map['modern'], label='ModernBERT'),
    Patch(facecolor=color_map['best'], label='Best Model (1024 tokens)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'fig3_model_comparison.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig3_model_comparison.pdf', bbox_inches='tight')
print(f"  💾 Saved: {output_dir}/fig3_model_comparison.png")
plt.close()
print()

# ============================================================================
# FIGURE 4: Architecture Impact on Hybrid Class
# ============================================================================
print("📊 Figure 4: Architecture Impact on Hybrid Class")

fig, ax = plt.subplots(figsize=(10, 5))

# Compare architectures on Hybrid class performance
arch_comparison = {
    'UniXcoder\n(RoBERTa-based)': {'overall': 0.8598, 'hybrid': 0.7730},
    'ModernBERT\n(510 tokens)': {'overall': 0.8632, 'hybrid': 0.8228},
    'ModernBERT\n(1024 tokens)': {'overall': 0.8712, 'hybrid': 0.8242}
}

x_pos = np.arange(len(arch_comparison))
width = 0.35

overall = [data['overall'] for data in arch_comparison.values()]
hybrid = [data['hybrid'] for data in arch_comparison.values()]

bars1 = ax.bar(x_pos - width/2, overall, width,
              label='Overall F1', color='#3498db', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, hybrid, width,
              label='Hybrid Class F1', color='#e74c3c', alpha=0.8)

# Add value labels and improvement percentages
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    # Overall F1 labels
    height1 = bar1.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1,
            f'{height1:.4f}',
            ha='center', va='bottom', fontsize=9)

    # Hybrid F1 labels
    height2 = bar2.get_height()
    ax.text(bar2.get_x() + bar2.get_width()/2., height2,
            f'{height2:.4f}',
            ha='center', va='bottom', fontsize=9)

    # Add improvement annotation for Hybrid
    if i > 0:
        prev_hybrid = list(arch_comparison.values())[0]['hybrid']
        improvement = ((height2 - prev_hybrid) / prev_hybrid) * 100
        ax.annotate(f'+{improvement:.1f}%',
                   xy=(x_pos[i] + width/2, height2),
                   xytext=(0, 15),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   color='#c0392b',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', alpha=0.8))

ax.set_ylabel('F1 Score')
ax.set_title('ModernBERT Architecture Impact on Hybrid Class Detection')
ax.set_xticks(x_pos)
ax.set_xticklabels(arch_comparison.keys())
ax.legend(loc='lower right')
ax.set_ylim([0.75, 0.90])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'fig4_hybrid_improvement.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig4_hybrid_improvement.pdf', bbox_inches='tight')
print(f"  💾 Saved: {output_dir}/fig4_hybrid_improvement.png")
plt.close()
print()

# ============================================================================
# FIGURE 5: Training Efficiency (Convergence Speed)
# ============================================================================
print("📊 Figure 5: Training Efficiency Comparison")

fig, ax = plt.subplots(figsize=(8, 5))

# Compare convergence speed
models_convergence = [
    {
        'path': 'models/unixcoder_full/checkpoint-24612/trainer_state.json',
        'name': 'UniXcoder (7 epochs)',
        'color': '#2ecc71',
        'final_f1': 0.8598
    },
    {
        'path': 'models/modernbert_base_full/checkpoint-10548/trainer_state.json',
        'name': 'ModernBERT (3 epochs)',
        'color': '#3498db',
        'final_f1': 0.8632
    },
    {
        'path': 'models/modernbert_longer_full/checkpoint-15625/trainer_state.json',
        'name': 'ModernBERT-1024 (5 epochs)',
        'color': '#e74c3c',
        'final_f1': 0.8712
    }
]

for model_info in models_convergence:
    try:
        with open(model_info['path'], 'r') as f:
            state = json.load(f)

        log_history = state['log_history']
        eval_entries = [entry for entry in log_history if 'eval_f1_macro' in entry]

        if eval_entries:
            # Normalize steps by total training steps to show epochs
            max_step = max(entry['step'] for entry in eval_entries)
            epochs = [entry['step'] / (max_step / state.get('epoch', 7)) for entry in eval_entries]
            f1_scores = [entry['eval_f1_macro'] for entry in eval_entries]

            ax.plot(epochs, f1_scores,
                   label=f"{model_info['name']} (Final: {model_info['final_f1']:.4f})",
                   color=model_info['color'],
                   linewidth=2.5,
                   marker='o',
                   markersize=5,
                   markevery=max(1, len(epochs)//8))
    except Exception as e:
        print(f"  ✗ {model_info['name']}: {e}")

ax.set_xlabel('Training Epoch')
ax.set_ylabel('Validation F1 Score (Macro)')
ax.set_title('Training Efficiency: Faster Convergence with Modern Architecture')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.75, 0.88])

plt.tight_layout()
plt.savefig(output_dir / 'fig5_training_efficiency.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig5_training_efficiency.pdf', bbox_inches='tight')
print(f"  💾 Saved: {output_dir}/fig5_training_efficiency.png")
plt.close()
print()

# ============================================================================
# Summary Table
# ============================================================================
print("📊 Creating Experimental Summary Table")

summary_data = {
    'Experiment': [
        'XGBoost (Baseline)',
        'RoBERTa-scratch',
        'CodeBERT',
        'UniXcoder',
        'Focal Loss',
        'ModernBERT (510)',
        'Class Weights',
        'Ensemble',
        'Random Crop',
        'ModernBERT (1024)',
        'ModernBERT (2048)*'
    ],
    'Test F1': [0.72, 0.7388, 0.8446, 0.8598, 0.8034, 0.8632, 0.8467, 0.8642, 0.8027, 0.8712, 0.88],
    'Hybrid F1': [0.55, 0.5419, 0.7607, 0.7730, 0.7629, 0.8228, 0.7904, 0.8025, 0.7571, 0.8242, 0.84],
    'Key Insight': [
        'Character n-grams',
        'Need pretraining',
        'Good baseline',
        'Best RoBERTa',
        'Failed (-5.6%)',
        'Architecture matters',
        'Failed (-1.3%)',
        'Marginal gain',
        'Failed (-6.0%)',
        'Best completed',
        'Promising (proj.)'
    ]
}

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
print()
df_summary.to_csv(output_dir / 'table_experimental_summary.csv', index=False)
print(f"  💾 Saved: {output_dir}/table_experimental_summary.csv")
print()

print("=" * 80)
print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
print("=" * 80)
print()
print("Output directory: report_figures/")
print()
print("Generated files:")
print("  • fig1_learning_curves.png/pdf - Learning curves for main models")
print("  • fig2_sequence_length_impact.png/pdf - Sequence length analysis")
print("  • fig3_model_comparison.png/pdf - All models comparison")
print("  • fig4_hybrid_improvement.png/pdf - Hybrid class breakthrough")
print("  • fig5_training_efficiency.png/pdf - Convergence speed comparison")
print("  • table_experimental_summary.csv - Summary table for report")
print()
print("✅ Ready for report writing!")
