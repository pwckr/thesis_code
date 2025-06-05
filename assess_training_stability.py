# %% script fo  running multiple training runs for the best configuration; for statistical fundament of training procedure
import os
from datetime import datetime

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd

from training_utils import (
    train_classification_model,
    evaluate_model,
)
from models import TCNModel
from data.dataset import DataProcessor
from data.preprocessing.data_registry import ENERCON_IDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results directory
results_dir = "/jobs/results/statistical_validation"
os.makedirs(results_dir, exist_ok=True)

def load_hyperparameters():
    """Load hyperparameters from JSON file for the best configuration."""
    json_filename = "best_hyperparams_long1_offset0_kmeans42.json"
    json_filepath = os.path.join("/jobs/results", json_filename)
    
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r') as f:
                data = json.load(f)
            print(f"Loaded hyperparameters from: {json_filename}")
            return data['best_hyperparameters']
        except Exception as e:
            print(f"Warning: Could not load hyperparameters from {json_filepath}: {e}")
    
    # Fallback to default hyperparameters for standard dataset
    print("Using default hyperparameters")
    return {
        'l': 0.0063,    # learning_rate from your thesis results
        'b': 256,       # batch_size
        'c': 16,        # channel_size  
        'k': 4,         # kernel_size
        'd': 0.391      # dropout
    }

def train_single_run(run_id, seed, hyperparams):
    """Train a model for a single run with a specific seed."""
    print(f"\nRun {run_id}/50 - Seed: {seed}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Fixed configuration - best performing from thesis
    long = True
    offset = 0
    kmeans = 42
    freq = "10min"
    window = 12
    ids = ENERCON_IDS
    
    # Load data
    dp = DataProcessor(ids=ids,
                      window_length=window,
                      offset=offset,
                      freq=freq,
                      kmeans=kmeans,
                      long_hist=long)
    
    train_sensors, train_codes, train_labels, val_sensors, val_codes, val_labels = dp.load_persisted_tensors()
    
    # Calculate statistics
    train_positive = train_labels.sum().item()
    train_negative = len(train_labels) - train_positive
    val_positive = val_labels.sum().item()
    val_negative = len(val_labels) - val_positive
    pos_weight = train_negative / train_positive if train_positive > 0 else 1.0
    
    # Prepare model
    num_sensors = train_sensors.shape[1]
    num_codes = train_codes.shape[1]
    
    train_sensor_mean = train_sensors.mean(dim=(0, 2))
    train_sensor_std = train_sensors.std(dim=(0, 2))
    train_sensor_std = torch.where(train_sensor_std == 0, 
                                  torch.ones_like(train_sensor_std), 
                                  train_sensor_std)
    
    # Create model with hyperparameters
    model = TCNModel(
        num_sensors=num_sensors,
        num_codes=num_codes,
        mean_sensors=train_sensor_mean.numpy(),
        std_sensors=train_sensor_std.numpy(),
        num_channels=[hyperparams['c'], hyperparams['c'], hyperparams['c']],
        kernel_size=hyperparams['k'],
        dropout=hyperparams['d']
    )
    
    # Train model
    _, loss_history = train_classification_model(
        model,
        train_sensors,
        train_codes,
        train_labels,
        val_sensors,
        val_codes,
        val_labels,
        pos_weight=torch.tensor([pos_weight]),
        learning_rate=hyperparams['l'],
        batch_size=hyperparams['b'],
        num_epochs=100,
        weight_decay=1e-5,
        patience=20,
    )
    
    # Evaluate model
    metrics = evaluate_model(model, val_sensors, val_codes, val_labels)
    
    # Return results
    return {
        'run_id': run_id,
        'seed': seed,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'binary_cross_entropy': metrics['binary_cross_entropy'],
        'average_precision': metrics['pr_curve']['average_precision'],
        'confusion_matrix': metrics['confusion_matrix'],
        'train_positive': train_positive,
        'train_negative': train_negative,
        'val_positive': val_positive,
        'val_negative': val_negative,
        'pos_weight': pos_weight,
        'final_train_loss': loss_history.get('train_loss', [])[-1] if loss_history.get('train_loss') else None,
        'final_val_loss': loss_history.get('val_loss', [])[-1] if loss_history.get('val_loss') else None,
        'min_val_loss': min(loss_history.get('val_loss', [1.0])) if loss_history.get('val_loss') else None,
        'epochs_trained': len(loss_history.get('val_loss', [])) if loss_history.get('val_loss') else None
    }

def plot_metric_distributions(results_df, save_dir):
    """Plot distributions of all metrics across runs."""
    metrics_to_plot = ['precision', 'recall']
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Create histogram with KDE
        sns.histplot(data=results_df, x=metric, kde=True, ax=ax, alpha=0.7)
        
        # Add mean line
        mean_val = results_df[metric].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.4f}')
        
        # Add std info
        std_val = results_df[metric].std()
        ax.text(0.05, 0.95, f'μ ± σ: {mean_val:.4f} ± {std_val:.4f}\nCV: {(std_val/mean_val)*100:.2f}%', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metric_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_stability(results_df, save_dir):
    """Plot training stability metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Precision vs Run ID (to check for trends)
    axes[0,0].scatter(results_df['run_id'], results_df['precision'], alpha=0.7)
    axes[0,0].axhline(results_df['precision'].mean(), color='red', linestyle='--', 
                     label=f'Mean: {results_df["precision"].mean():.4f}')
    axes[0,0].set_xlabel('Run ID')
    axes[0,0].set_ylabel('Precision')
    axes[0,0].set_title('Precision Across Training Runs')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Training Loss Distribution
    if 'final_train_loss' in results_df.columns and results_df['final_train_loss'].notna().sum() > 0:
        sns.histplot(data=results_df, x='final_train_loss', kde=True, ax=axes[0,1], alpha=0.7)
        axes[0,1].set_title('Final Training Loss Distribution')
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Validation Loss Distribution  
    if 'min_val_loss' in results_df.columns and results_df['min_val_loss'].notna().sum() > 0:
        sns.histplot(data=results_df, x='min_val_loss', kde=True, ax=axes[1,0], alpha=0.7)
        axes[1,0].set_title('Minimum Validation Loss Distribution')
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Precision vs Recall scatter
    axes[1,1].scatter(results_df['recall'], results_df['precision'], alpha=0.7)
    axes[1,1].set_xlabel('Recall')
    axes[1,1].set_ylabel('Precision')
    axes[1,1].set_title('Precision vs Recall')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run multiple training iterations."""
    print("Starting statistical validation with 50 training runs...")
    print("Configuration: Standard Dataset (long=True), offset=0, kmeans=42, freq=10min")
    
    # Load hyperparameters
    hyperparams = load_hyperparameters()
    print(f"Using hyperparameters: {hyperparams}")
    
    # Generate seeds for reproducibility
    base_seed = 42
    seeds = [base_seed + i * 137 for i in range(50)]  # Use different seeds
    
    # Store all results
    all_results = []
    
    # Run training 50 times
    for i in range(50):
        try:
            result = train_single_run(i + 1, seeds[i], hyperparams)
            all_results.append(result)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/50 runs")
                # Print running statistics
                temp_df = pd.DataFrame(all_results)
                print(f"Running mean precision: {temp_df['precision'].mean():.4f} ± {temp_df['precision'].std():.4f}")
                print(f"Running mean recall: {temp_df['recall'].mean():.4f} ± {temp_df['recall'].std():.4f}")
                
        except Exception as e:
            print(f"ERROR in run {i + 1}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate comprehensive statistics
    print(f"\n{'='*80}")
    print("STATISTICAL VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Successfully completed: {len(all_results)}/50 runs")
    print(f"Configuration: TCN with K-means clustering (K=42), Standard Dataset, 10-min horizon")
    
    # Main metrics statistics
    metrics = ['precision', 'recall', 'f1', 'auc', 'binary_cross_entropy', 'average_precision']
    
    print(f"\nMETRICS SUMMARY:")
    print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'CV(%)':<8} {'Min':<10} {'Max':<10} {'95% CI':<20}")
    print("-" * 90)
    
    for metric in metrics:
        values = results_df[metric]
        mean_val = values.mean()
        std_val = values.std()
        cv = (std_val / mean_val) * 100
        min_val = values.min()
        max_val = values.max()
        
        # 95% confidence interval
        ci_lower = mean_val - 1.96 * (std_val / np.sqrt(len(values)))
        ci_upper = mean_val + 1.96 * (std_val / np.sqrt(len(values)))
        
        print(f"{metric:<20} {mean_val:<10.4f} {std_val:<10.4f} {cv:<8.2f} {min_val:<10.4f} {max_val:<10.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Compare to thesis reported values
    print(f"\nCOMPARISON TO THESIS REPORTED VALUES:")
    print(f"Thesis reported precision: 0.5881")
    print(f"Statistical validation precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"Thesis reported recall: 0.5209") 
    print(f"Statistical validation recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    
    # Check if thesis values are within confidence intervals
    precision_ci = [results_df['precision'].mean() - 1.96 * results_df['precision'].std()/np.sqrt(len(results_df)),
                    results_df['precision'].mean() + 1.96 * results_df['precision'].std()/np.sqrt(len(results_df))]
    
    recall_ci = [results_df['recall'].mean() - 1.96 * results_df['recall'].std()/np.sqrt(len(results_df)),
                 results_df['recall'].mean() + 1.96 * results_df['recall'].std()/np.sqrt(len(results_df))]
    
    print(f"\nVALIDATION CHECK:")
    print(f"Thesis precision (0.5881) within 95% CI [{precision_ci[0]:.4f}, {precision_ci[1]:.4f}]: {precision_ci[0] <= 0.5881 <= precision_ci[1]}")
    print(f"Thesis recall (0.5209) within 95% CI [{recall_ci[0]:.4f}, {recall_ci[1]:.4f}]: {recall_ci[0] <= 0.5209 <= recall_ci[1]}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSV with all results
    csv_path = os.path.join(results_dir, f'statistical_validation_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Save summary statistics
    summary_path = os.path.join(results_dir, f'statistical_validation_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL VALIDATION SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration: TCN with K-means clustering (K=42)\n")
        f.write(f"Dataset: Standard Dataset (long=True, freq=10min)\n")
        f.write(f"Prediction Horizon: 10 minutes (offset=0)\n")
        f.write(f"Number of training runs: {len(all_results)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("HYPERPARAMETERS:\n")
        f.write("-" * 40 + "\n")
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("STATISTICAL SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'CV(%)':<8} {'95% CI':<20}\n")
        f.write("-" * 70 + "\n")
        
        for metric in metrics:
            values = results_df[metric]
            mean_val = values.mean()
            std_val = values.std()
            cv = (std_val / mean_val) * 100
            ci_lower = mean_val - 1.96 * (std_val / np.sqrt(len(values)))
            ci_upper = mean_val + 1.96 * (std_val / np.sqrt(len(values)))
            f.write(f"{metric:<20} {mean_val:<10.4f} {std_val:<10.4f} {cv:<8.2f} [{ci_lower:.4f}, {ci_upper:.4f}]\n")
        
        f.write(f"\nKEY FINDINGS:\n")
        f.write(f"- Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}\n")
        f.write(f"- Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}\n")
        f.write(f"- F1-Score: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}\n")
        f.write(f"- Coefficient of Variation (Precision): {(results_df['precision'].std()/results_df['precision'].mean())*100:.2f}%\n")
        f.write(f"- Results are {'stable' if (results_df['precision'].std()/results_df['precision'].mean())*100 < 5 else 'moderately variable'}\n")
    
    # Create plots
    plot_metric_distributions(results_df, results_dir)
    plot_training_stability(results_df, results_dir)
    
    print(f"Summary saved to: {summary_path}")
    print(f"Plots saved to: {results_dir}")
    print(f"\nStatistical validation complete!")

if __name__ == "__main__":
    main()