"""Utility functions for TCN trainings."""
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score



def train_classification_model(
    model, 
    train_sensors, 
    train_codes, 
    train_labels,
    val_sensors,
    val_codes,
    val_labels,
    pos_weight,
    learning_rate=0.005,
    batch_size=64,
    num_epochs=20,
    weight_decay=1e-5,
    patience=5,       # Early Stopping
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_size = len(train_sensors)
    val_size = len(val_sensors)
    n_batches = (train_size + batch_size - 1) // batch_size
    best_val_loss = float('inf')
    no_improve_epochs = 0
    loss_history = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        indices = torch.randperm(train_size)
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_size)
            batch_indices = indices[start_idx:end_idx]
            
            sensor_batch = train_sensors[batch_indices].to(device)
            code_batch = train_codes[batch_indices].to(device)
            label_batch = train_labels[batch_indices].to(device)
            
            optimizer.zero_grad()
            outputs = model(sensor_batch, code_batch)
            loss = criterion(outputs, label_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item() * len(batch_indices)
            
    
        model.eval()
        val_loss = 0.0

        
        with torch.no_grad():
            val_batches = (val_size + batch_size - 1) // batch_size
            for i in range(val_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, val_size)
                
                val_sensor_batch = val_sensors[start_idx:end_idx].to(device)
                val_code_batch = val_codes[start_idx:end_idx].to(device)
                val_label_batch = val_labels[start_idx:end_idx].to(device)
                
                val_outputs = model(val_sensor_batch, val_code_batch)
                val_batch_loss = criterion(val_outputs, val_label_batch)
                val_loss += val_batch_loss.item() * (end_idx - start_idx)
                
        
        train_loss /= train_size
        val_loss /= val_size
        loss_history['train'].append(train_loss)
        loss_history['val'].append(val_loss)
        
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Metric: {train_loss:.4f}, '
              f'Val Metric: {val_loss:.4f}')


        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'best_model.pt')

        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'Early stopping nach {epoch+1} Epochen')
                break
    
    # Bestes Modell laden
    model.load_state_dict(torch.load('best_model.pt'))
    
    return model, loss_history


def plot_training(train, val):
    epochs = range(1, len(train) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val, 'r-', label='Validation Loss', linewidth=2)
    
    # Find minimum validation loss and mark it
    min_val_idx = np.argmin(val)
    min_val_epoch = min_val_idx + 1  # +1 because epochs start from 1
    min_val_loss = val[min_val_idx]
    
    plt.plot(min_val_epoch, min_val_loss, 'ro', markersize=10, 
             label=f'Best Val Loss (Epoch {min_val_epoch})')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, sensors, codes, labels, device=None, compute_pr_curve=True, save_pr_curve=False, pr_curve_path=None):
    """
    Evaluates a model on the given data and returns detailed metrics including PR curve and binary cross-entropy.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        sensors (torch.Tensor): Sensor input data
        codes (torch.Tensor): Code input data
        labels (torch.Tensor): Ground truth labels
        device (torch.device, optional): Device for computation
        compute_pr_curve (bool): Whether to compute precision-recall curve
        save_pr_curve (bool): Whether to save the PR curve to a file
        pr_curve_path (str, optional): Path to save the PR curve
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    with torch.no_grad():
        sensors = sensors.to(device)
        codes = codes.to(device)
        labels = labels.to(device)
        
        # Get predictions
        predictions = model(sensors, codes)
        binary_preds = (predictions > 0.5).float()
        
        # Calculate binary cross-entropy
        bce_loss_fn = torch.nn.BCELoss()
        # Handle any edge cases where predictions might be exactly 0 or 1
        eps = 1e-7
        predictions_safe = torch.clamp(predictions, eps, 1 - eps)
        bce = bce_loss_fn(predictions_safe, labels).item()
        
        # Move tensors to CPU for sklearn metrics
        labels_np = labels.cpu().numpy()
        preds_np = binary_preds.cpu().numpy()
        probs_np = predictions.cpu().numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds_np, average='binary', zero_division=0
        )
        
        # Calculate AUC, handling edge cases
        try:
            auc = roc_auc_score(labels_np, probs_np)
        except:
            auc = float('nan')
        
        # Confusion matrix
        cm = confusion_matrix(labels_np, preds_np)
        
        # Base metrics dictionary
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": float(auc),
            "binary_cross_entropy": bce,  # Added BCE metric
            "confusion_matrix": cm.tolist()
        }
        
        # Compute precision-recall curve if requested
        if compute_pr_curve:
            # Compute precision-recall pairs for different probability thresholds
            pr_precision, pr_recall, thresholds = precision_recall_curve(labels_np, probs_np)
            
            # Calculate average precision score
            ap = average_precision_score(labels_np, probs_np)
            
            # Add PR curve data to metrics
            metrics.update({
                "pr_curve": {
                    "precision": pr_precision.tolist(),
                    "recall": pr_recall.tolist(),
                    "thresholds": thresholds.tolist() if len(thresholds) > 0 else [],
                    "average_precision": float(ap)
                }
            })
            
            # Save PR curve plot if requested
            if save_pr_curve:
                if pr_curve_path is None:
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                    pr_curve_path = f"pr_curve_{timestamp}.png"
                
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(pr_curve_path) if os.path.dirname(pr_curve_path) else '.', exist_ok=True)
                
                # Plot and save PR curve
                plot_precision_recall_curve(pr_precision, pr_recall, ap, save_path=pr_curve_path, show=True)
                print(f"PR curve saved to: {pr_curve_path}")
        
        return metrics


def plot_precision_recall_curve(precision, recall, average_precision, save_path=None, show=True):
    """Create PR Curve, can be saved or shown"""
    import seaborn as sns
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 8))
    
    # Plot the precision-recall curve with seaborn styling
    ax = plt.gca()
    
    # Fill the area under PR curve
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    
    # Plot the PR curve
    sns.lineplot(x=recall, y=precision, color='blue', linewidth=2.5)

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curve (AP={average_precision:.3f})', fontsize=16, fontweight='bold')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    
    plt.tight_layout()
    
    # Save if path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return plt.gcf()

