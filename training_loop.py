# %% Extensive script that was run on the cluster for getting tcn training results
import os
import re
from datetime import datetime

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from training_utils import (
    train_classification_model,
    evaluate_model,
    plot_precision_recall_curve
)
from models import TCNModel
from data.dataset import DataProcessor
from data.preprocessing.data_registry import ENERCON_IDS, HIGH_RES_IDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create results directory
results_dir = "/jobs/results"
os.makedirs(results_dir, exist_ok=True)

def load_hyperparameters(long, offset, kmeans):
    """Load hyperparameters from JSON file - use offset=0 results for all offsets."""
    # Always load from offset=0 file (where HPO was performed)
    json_filename = f"best_hyperparams_long{long}_offset0_kmeans{kmeans}.json"
    json_filepath = os.path.join(results_dir, json_filename)
    
    if os.path.exists(json_filepath):
        try:
            with open(json_filepath, 'r') as f:
                data = json.load(f)
            print(f"Loaded hyperparameters from: {json_filename}")
            return data['best_hyperparameters']
        except Exception as e:
            print(f"Warning: Could not load hyperparameters from {json_filepath}: {e}")
    else:
        print(f"Warning: Hyperparameter file not found: {json_filepath}")


def convert_to_timedelta_string(freq, value):
    """
    Convert frequency and value to human-readable timedelta string.
    
    Args:
        freq (str): Frequency string like "10min" or "1min"
        value (int): Number of periods
        
    Returns:
        str: Human-readable timedelta string
    """
    # Parse frequency string to extract number and unit
    freq_match = re.match(r'(\d+)(\w+)', freq)
    if not freq_match:
        return f"{value} periods"
    
    freq_number = int(freq_match.group(1))
    freq_unit = freq_match.group(2)
    
    # Calculate total minutes
    if freq_unit in ['min', 'minute', 'minutes']:
        total_minutes = freq_number * value
    elif freq_unit in ['h', 'hour', 'hours']:
        total_minutes = freq_number * 60 * value
    elif freq_unit in ['s', 'sec', 'second', 'seconds']:
        total_minutes = (freq_number * value) / 60
    else:
        return f"{value} periods"
    
    # Convert to appropriate unit
    if total_minutes < 60:
        if total_minutes == int(total_minutes):
            return f"{int(total_minutes)}min"
        else:
            return f"{total_minutes:.1f}min"
    elif total_minutes < 1440:  # Less than 24 hours
        hours = total_minutes / 60
        if hours == int(hours):
            return f"{int(hours)}h"
        else:
            return f"{hours:.1f}h"
    else:  # Days
        days = total_minutes / 1440
        if days == int(days):
            return f"{int(days)}d"
        else:
            return f"{days:.1f}d"

def plot_precision_recall_curve(precision, recall, average_precision, config_info, save_path=None, show=True):
    """
    Plot precision-recall curve with configuration information
    
    Args:
        precision (array): Precision values
        recall (array): Recall values
        average_precision (float): Average precision score
        config_info (dict): Configuration information for the plot
        save_path (str, optional): Path to save the figure
        show (bool): Whether to show the plot
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 8))
    
    # Plot the precision-recall curve with seaborn styling
    ax = plt.gca()
    
    # Fill the area under PR curve
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    
    # Plot the PR curve
    sns.lineplot(x=recall, y=precision, color='blue', linewidth=2.5)
    
    # Add baseline
    plt.plot([0, 1], [precision[0], precision[0]], linestyle='--', color='gray', alpha=0.8, label='Baseline')
    
    # Customize plot
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    
    # Enhanced title with configuration - Updated naming
    dataset_type = "Standard Sensors" if config_info['long'] else "High-Res Sensors"
    title = f'Precision-Recall Curve - {dataset_type}\n'
    
    plot_offset = config_info['offset'] if config_info['offset'] == 10 else config_info['offset'] + 1
    
    offset_str = convert_to_timedelta_string(config_info['freq'], plot_offset)
    window_str = convert_to_timedelta_string(config_info['freq'], config_info['window'])
    
    if config_info['kmeans'] is None:
        kmeans_str = "OneHot Status-Codes"
    else:
        kmeans_str = f"Number of Status-Code-Clusters: {config_info['kmeans']}"
    
    title += f'Offset: {offset_str}, Window: {window_str}'
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add configuration and performance text - Updated format
    textstr = f'Average Precision (AP): {average_precision:.3f}\n'
    textstr += f'Dataset: {dataset_type}\n'
    textstr += f'Offset: {offset_str}\n'
    textstr += f'Window: {window_str}\n'
    textstr += f'{kmeans_str}\n'
    textstr += f'Frequency: {config_info["freq"]}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.4, textstr, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return plt.gcf()

def save_training_results(config, hyperparams, metrics, loss_history, save_path):
    """Save training results to a text file."""
    train_losses = loss_history["train"]
    val_losses = loss_history["val"]

    # Calculate training stats if we have the data
    if len(val_losses) > 0:
        min_val_epoch = np.argmin(val_losses) + 1
        min_val_loss = min(val_losses)
        final_train_loss = train_losses[-1] if len(train_losses) > 0 else "N/A"
        total_epochs = len(val_losses)
    else:
        min_val_epoch = "N/A"
        min_val_loss = "N/A"
        final_train_loss = "N/A"
        total_epochs = "N/A"
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset Type: {'Standard Sensors (long_hist=True)' if config['long'] else 'High-Res Sensors (long_hist=False)'}\n")
        f.write(f"Offset: {config['offset']}\n")
        f.write(f"K-means: {config['kmeans']}\n")
        f.write(f"Window Length: {config['window']}\n")
        f.write(f"Frequency: {config['freq']}\n")
        f.write(f"Positive Class Weight: {config['pos_weight']:.4f}\n\n")
        
        # Hyperparameters
        f.write("HYPERPARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Learning Rate: {hyperparams['l']:.6f}\n")
        f.write(f"Batch Size: {hyperparams['b']}\n")
        f.write(f"Channel Size: {hyperparams['c']}\n")
        f.write(f"Kernel Size: {hyperparams['k']}\n")
        f.write(f"Dropout: {hyperparams['d']:.4f}\n")
        f.write(f"HPO Source: JSON file (offset=0 hyperparameters)\n\n")
        
        # Training Information
        f.write("TRAINING INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Epochs: {total_epochs}\n")
        f.write(f"Best Model Selected at Epoch: {min_val_epoch}\n")
        f.write(f"Minimum Validation Loss: {min_val_loss}\n")
        f.write(f"Final Training Loss: {final_train_loss}\n\n")
        
        # Performance Metrics
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"ROC AUC: {metrics['auc']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n")
        f.write(f"Binary Cross-Entropy: {metrics['binary_cross_entropy']:.4f}\n")
        f.write(f"Average Precision: {metrics['pr_curve']['average_precision']:.4f}\n\n")
        
        # Confusion Matrix
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 40 + "\n")
        cm = metrics["confusion_matrix"]
        f.write(f"True Negatives:  {cm[0][0]:6d}    False Positives: {cm[0][1]:6d}\n")
        f.write(f"False Negatives: {cm[1][0]:6d}    True Positives:  {cm[1][1]:6d}\n\n")
        
        # Dataset Statistics
        f.write("DATASET STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Positive Samples: {config['train_positive']}\n")
        f.write(f"Training Negative Samples: {config['train_negative']}\n")
        f.write(f"Validation Positive Samples: {config['val_positive']}\n")
        f.write(f"Validation Negative Samples: {config['val_negative']}\n")
        
        f.write("="*80 + "\n")
    
    print(f"Training results saved to: {save_path}")

def train_single_configuration(long, offset, kmeans):
    """Train a model for a single configuration."""
    print(f"\n{'='*60}")
    print(f"TRAINING: long_hist={long}, offset={offset}, kmeans={kmeans}")
    print(f"{'='*60}")
    
    # Set up data configuration
    if long:
        ids = ENERCON_IDS
        freq = "10min"
        window = 12
    else:
        ids = HIGH_RES_IDS
        freq = "1min"
        window = 120
    
    # Load hyperparameters
    hyperparams = load_hyperparameters(long, offset, kmeans)
    print(f"Using hyperparameters: {hyperparams}")
    
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
    
    print(f"{'Standard Sensors' if long else 'High-Res Sensors'} Dataset - freq: {freq}, window: {window}, offset: {offset}")
    print(f"Training: {train_positive} positive, {train_negative} negative samples")
    print(f"Validation: {val_positive} positive, {val_negative} negative samples")
    print(f"Positive class weight: {pos_weight:.2f}")
    
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
    print("Starting training...")
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
    
    # Print results
    print(f"\nRESULTS:")
    print(f"ROC AUC: {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Binary Cross-Entropy: {metrics['binary_cross_entropy']:.4f}")
    print(f"Average Precision: {metrics['pr_curve']['average_precision']:.4f}")
    
    # Prepare file names
    base_filename = f"training_long{long}_offset{offset}_kmeans{kmeans}"
    
    # Save precision-recall curve
    pr_path = os.path.join(results_dir, f"{base_filename}_pr_curve.png")
    config_info = {
        'long': long, 'offset': offset, 'kmeans': kmeans,
        'window': window, 'freq': freq
    }
    plot_precision_recall_curve(
        metrics['pr_curve']['precision'],
        metrics['pr_curve']['recall'], 
        metrics['pr_curve']['average_precision'],
        config_info,
        save_path=pr_path,
        show=False
    )
    
    # Save results text file
    results_path = os.path.join(results_dir, f"{base_filename}_results.txt")
    config = {
        'long': long, 'offset': offset, 'kmeans': kmeans,
        'window': window, 'freq': freq, 'pos_weight': pos_weight,
        'train_positive': train_positive, 'train_negative': train_negative,
        'val_positive': val_positive, 'val_negative': val_negative
    }
    save_training_results(config, hyperparams, metrics, loss_history, results_path)
    
    return metrics, loss_history

def main():
    """Main training loop over all configurations."""
    print("Starting comprehensive training across all configurations...")
    
    # Define all configurations
    configurations = []
    
    # For long=True: offsets 0, 1, 5, 8, 11
    for kmeans in [None, 42]:
        for offset in [0, 1, 5, 8, 11]:
            configurations.append((True, offset, kmeans))
    
    # For long=False: offsets 0, 10, 19, 29, 59, 89, 119  
    for kmeans in [None, 42]:
        for offset in [0, 10, 19, 29, 59, 89, 119]:
            configurations.append((False, offset, kmeans))
    
    print(f"Total configurations to train: {len(configurations)}")
    
    # Store all results
    all_results = []
    
    # Train each configuration
    for i, (long, offset, kmeans) in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] Training configuration: long={long}, offset={offset}, kmeans={kmeans}")
        
        try:
            metrics, loss_history = train_single_configuration(long, offset, kmeans)
            all_results.append({
                'long': long, 'offset': offset, 'kmeans': kmeans,
                'metrics': metrics, 'loss_history': loss_history
            })
            
        except Exception as e:
            print(f"ERROR in configuration long={long}, offset={offset}, kmeans={kmeans}: {e}")
            continue
    
    # Save summary of all results
    summary_path = os.path.join(results_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING SUMMARY - ALL CONFIGURATIONS\n")
        f.write("="*80 + "\n")
        f.write(f"Total configurations trained: {len(all_results)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Custom sorting: Group by (long, kmeans) combination, then sort by precision within each group
        def sort_key(result):
            long = result['long']
            kmeans = result['kmeans']
            precision = result['metrics']['precision']
            
            if long and kmeans is None:
                group = 0
            elif not long and kmeans is None:
                group = 1
            elif long and kmeans == 42:
                group = 2
            else:  # not long and kmean == 42
                group = 3
            
            # Within each group, sort by precision (descending)
            return (group, -precision)
        
        sorted_results = sorted(all_results, key=sort_key)
        
        f.write("RESULTS GROUPED BY CONFIGURATION (ranked by precision within each group):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<5} {'Long':<6} {'Offset':<7} {'K-means':<8} {'Precision':<10} {'F1':<7} {'AUC':<7} {'Recall':<7}\n")
        f.write("-" * 80 + "\n")
        
        current_group = -1
        rank_in_group = 0
        
        for result in sorted_results:
            long = result['long']
            kmeans = result['kmeans']
            
            # Determine group
            if long and kmeans is None:
                group = 0
                group_name = "Long=True, K-means=None"
            elif not long and kmeans is None:
                group = 1
                group_name = "Long=False, K-means=None"
            elif long and kmeans == 42:
                group = 2
                group_name = "Long=True, K-means=42"
            else:
                group = 3
                group_name = "Long=False, K-means=42"
            
            # Add group separator
            if group != current_group:
                if current_group != -1:  # Not the first group
                    f.write("-" * 80 + "\n")
                f.write(f"\n{group_name.upper()}:\n")
                f.write("-" * 80 + "\n")
                current_group = group
                rank_in_group = 0
            
            rank_in_group += 1
            m = result['metrics']
            f.write(f"{rank_in_group:<5} {result['long']:<6} {result['offset']:<7} {str(result['kmeans']):<8} "
                   f"{m['precision']:<10.4f} {m['f1']:<7.4f} {m['auc']:<7.4f} {m['recall']:<7.4f}\n")
    
    print(f"\nTraining complete! Summary saved to: {summary_path}")
    print(f"Trained {len(all_results)} configurations successfully.")

if __name__ == "__main__":
    main()