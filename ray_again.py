"""Minimal HPO for neural network using Ray - Get best hyperparameters only."""

import os
import torch
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import ray
from datetime import datetime

from models import TCNModel
from data.dataset import DataProcessor
from data.preprocessing.data_registry import HIGH_RES_IDS, ENERCON_IDS

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

# we need a batch loop function so that we can report metrics to ray for each epoch;
# this is not possible for methods developed and used elsewhere
def batch_loop(sensors, codes, labels, batch_size, criterion, optimizer, model):
    """Training loop for one epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    train_size = len(sensors)
    n_batches = (train_size + batch_size - 1) // batch_size
    total_loss = 0.0
    
    # Shuffle indices
    indices = torch.randperm(train_size)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, train_size)
        batch_indices = indices[start_idx:end_idx]
        
        sensor_batch = sensors[batch_indices].to(device)
        code_batch = codes[batch_indices].to(device)
        label_batch = labels[batch_indices].to(device)
        
        optimizer.zero_grad()
        outputs = model(sensor_batch, code_batch)
        loss = criterion(outputs, label_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch_indices)
    
    return total_loss / train_size


def evaluate_model_bce(model, sensors, codes, labels, pos_weight):
    """Evaluate model and return BCE loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        sensors_dev = sensors.to(device)
        codes_dev = codes.to(device)
        labels_dev = labels.to(device)
        
        predictions = model(sensors_dev, codes_dev)
        
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        val_loss = criterion(predictions, labels_dev).item()
    
    return val_loss


def t(config: dict, data_dict: dict): # short name due to windows errors concerning paths of result-file that are too long for windows
    """Minimal training function - just train and report final result."""
    
    # Extract data
    train_sensors = data_dict["train_sensors"]
    train_codes = data_dict["train_codes"] 
    train_labels = data_dict["train_labels"]
    val_sensors = data_dict["val_sensors"]
    val_codes = data_dict["val_codes"]
    val_labels = data_dict["val_labels"]
    pos_weight = data_dict["pos_weight"]
    num_sensors = data_dict["num_sensors"]
    num_codes = data_dict["num_codes"]
    max_epochs = data_dict["max_epochs"]
    train_sensor_mean = data_dict["train_sensor_mean"]
    train_sensor_std = data_dict["train_sensor_std"]
    
    # Extract hyperparameters
    learning_rate = config["l"]
    batch_size = config["b"]
    channel_size = config["c"]
    kernel_size = config["k"]
    dropout = config["d"]
    
    # Create model with all hyperparameters including normalization
    model = TCNModel(
        num_sensors=num_sensors,
        num_codes=num_codes,
        mean_sensors=train_sensor_mean,
        std_sensors=train_sensor_std,
        num_channels=[channel_size, channel_size, channel_size],
        kernel_size=kernel_size,
        dropout=dropout
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Simple training loop - report periodically for ASHA scheduler
    for epoch in range(1, max_epochs + 1):
        train_loss = batch_loop(
            train_sensors, train_codes, train_labels, 
            batch_size, criterion, optimizer, model
        )
        
        val_loss = evaluate_model_bce(model, val_sensors, val_codes, val_labels, pos_weight)
        
        # Report to scheduler (allows early stopping of bad trials)
        tune.report({"BCE_Loss": val_loss})


# Hyperparameter search space
CONFIG = {
    "l": tune.loguniform(1e-4, 1e-2),
    "b": tune.choice([32, 64, 128, 256]),
    "c": tune.choice([16, 24, 32, 48]),
    "k": tune.choice([2, 3, 4, 5, 7]),
    "d": tune.uniform(0.0, 0.5)
}


def save_results_to_file(result, long, offset, kmeans, num_samples, max_num_epochs):
    """Save all HPO results to a text file."""
    
    # Create results directory if it doesn't exist
    # For K8s, use mounted volume path
    results_dir = os.environ.get("RESULTS_DIR", "/jobs/results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with configuration parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hpo_results_long{long}_offset{offset}_kmeans{kmeans}_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RAY HPO RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration Parameters:\n")
        f.write(f"  - long_hist: {long}\n")
        f.write(f"  - offset: {offset}\n")
        f.write(f"  - kmeans: {kmeans}\n")
        f.write(f"  - num_samples: {num_samples}\n")
        f.write(f"  - max_epochs: {max_num_epochs}\n")
        f.write("="*80 + "\n\n")
        
        try:
            # Try to get best result first
            best_result = result.get_best_result("BCE_Loss", "min")
            
            if best_result is not None:
                f.write("BEST CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Validation BCE Loss: {best_result.metrics['BCE_Loss']:.6f}\n")
                f.write("Hyperparameters:\n")
                for param, value in best_result.config.items():
                    f.write(f"  {param:15}: {value}\n")
                f.write("\n")
                
                # Try to get all results if available
                try:
                    all_results = result.get_dataframe()
                    if 'BCE_Loss' in all_results.columns:
                        all_results = all_results.sort_values('BCE_Loss')
                        f.write(f"TOTAL TRIALS COMPLETED: {len(all_results)}\n\n")
                        
                        # Write all results
                        f.write("ALL TRIAL RESULTS (sorted by BCE Loss):\n")
                        f.write("-" * 40 + "\n")
                        for idx, (_, row) in enumerate(all_results.iterrows(), 1):
                            f.write(f"Trial {idx:2d} - BCE Loss: {row['BCE_Loss']:.6f}\n")
                            f.write(f"  Learning Rate (l): {row['config/l']:.6f}\n")
                            f.write(f"  Batch Size (b):    {row['config/b']}\n")
                            f.write(f"  Channel Size (c):  {row['config/c']}\n")
                            f.write(f"  Kernel Size (k):   {row['config/k']}\n")
                            f.write(f"  Dropout (d):       {row['config/d']:.4f}\n")
                            f.write("\n")
                        
                        # Summary statistics
                        f.write("SUMMARY STATISTICS:\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Best BCE Loss:     {all_results['BCE_Loss'].min():.6f}\n")
                        f.write(f"Worst BCE Loss:    {all_results['BCE_Loss'].max():.6f}\n")
                        f.write(f"Mean BCE Loss:     {all_results['BCE_Loss'].mean():.6f}\n")
                        f.write(f"Std BCE Loss:      {all_results['BCE_Loss'].std():.6f}\n")
                    else:
                        f.write("WARNING: BCE_Loss column not found in results dataframe\n")
                        f.write(f"Available columns: {list(all_results.columns)}\n")
                except Exception as e:
                    f.write(f"WARNING: Could not retrieve full results dataframe: {e}\n")
                    
            else:
                f.write("ERROR: No best result found - all trials may have failed\n")
                
        except Exception as e:
            f.write(f"ERROR: Could not retrieve results: {e}\n")
            f.write("This likely means all trials failed or results weren't properly saved.\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to: {filepath}")
    
    # Also save best hyperparameters as JSON
    json_filename = f"best_hyperparams_long{long}_offset{offset}_kmeans{kmeans}.json"
    json_filepath = os.path.join(results_dir, json_filename)
    
    try:
        best_result = result.get_best_result("BCE_Loss", "min")
        if best_result is not None:
            # Prepare JSON data
            json_data = {
                "configuration": {
                    "long_hist": long,
                    "offset": offset,
                    "kmeans": kmeans,
                    "num_samples": num_samples,
                    "max_epochs": max_num_epochs
                },
                "best_hyperparameters": best_result.config,
                "best_validation_loss": best_result.metrics['BCE_Loss'],
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save JSON file
            import json
            with open(json_filepath, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)
            
            print(f"Best hyperparameters saved to: {json_filepath}")
        else:
            print("WARNING: Could not save JSON - no best result found")
            
    except Exception as e:
        print(f"WARNING: Could not save JSON file: {e}")
    
    return filepath


def main(num_samples: int = 20, max_num_epochs: int = 50):
    """Run HPO and return best hyperparameters and metric."""
    
    # Initialize Ray - support both local and cluster modes
    if not ray.is_initialized():
        ray_address = os.environ.get("RAY_ADDRESS", None)
        if ray_address:
            print(f"Connecting to Ray cluster at: {ray_address}")
            ray.init(address=ray_address)
        else:
            print("Starting local Ray instance")
            ray.init()
    
    # Load data
    long = True
    offset = 0
    kmeans = 42
    
    if long:
        ids = ENERCON_IDS
        freq = "10min"
        window = 12
    else:
        ids = HIGH_RES_IDS
        freq = "1min"
        window = 120

    print("Loading data...")
    dp = DataProcessor(ids=ids,
                       window_length=window,
                       offset=offset, 
                       freq=freq,
                       long_hist=long,
                       kmeans=kmeans,)
    
    train_sensors, train_codes, train_labels, val_sensors, val_codes, val_labels = dp.load_persisted_tensors()
    
    # Calculate normalization statistics from training data
    train_sensor_mean = train_sensors.mean(dim=(0, 2))
    train_sensor_std = train_sensors.std(dim=(0, 2))

    # Handle zero standard deviation (constant features)
    train_sensor_std = torch.where(train_sensor_std == 0, 
                                  torch.ones_like(train_sensor_std), 
                                  train_sensor_std)
    
    # Calculate class weights
    train_positive = train_labels.sum().item()
    train_negative = len(train_labels) - train_positive
    val_positive = val_labels.sum().item()
    val_negative = len(val_labels) - val_positive
    pos_weight = train_negative / train_positive if train_positive > 0 else 1.0
    
    print(f"Dataset: {train_positive} positive, {train_negative} negative training samples")
    print(f"Validation: {val_positive} positive, {val_negative} samples")
    print(f"Positive class weight: {pos_weight:.2f}")
    print(f"Sensor normalization - Mean range: [{train_sensor_mean.min():.4f}, {train_sensor_mean.max():.4f}]")
    print(f"Sensor normalization - Std range: [{train_sensor_std.min():.4f}, {train_sensor_std.max():.4f}]")
    
    # Prepare data for Ray
    data_dict = {
        "train_sensors": train_sensors,
        "train_codes": train_codes,
        "train_labels": train_labels,
        "val_sensors": val_sensors,
        "val_codes": val_codes,
        "val_labels": val_labels,
        "pos_weight": pos_weight,
        "num_sensors": train_sensors.shape[1],
        "num_codes": train_codes.shape[1],
        "window": window,
        "max_epochs": max_num_epochs,
        "train_sensor_mean": train_sensor_mean.numpy(),  # Convert to numpy for the model
        "train_sensor_std": train_sensor_std.numpy(),    # Convert to numpy for the model
    }
    
    print(f"Starting HPO with {num_samples} trials...")
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="BCE_Loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2,
    )
    
    # Run optimization with proper storage
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(t, data_dict=data_dict),
            resources={"cpu": 3, "gpu": .5}
        ),
        param_space=CONFIG,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
        ),
        run_config=train.RunConfig(
            name="tcn_hpo",
            log_to_file=False,
            verbose=1,
            # Enable storage to a temporary directory for result collection
            storage_path="/tmp/ray_results",
            # Keep minimal logging
            progress_reporter=tune.CLIReporter(metric_columns=["BCE_Loss"])
        )
    )
    
    result = tuner.fit()
    
    # Save all results to file
    save_results_to_file(result, long, offset, kmeans, num_samples, max_num_epochs)
    
    # Extract and print the best results
    best_result = result.get_best_result("BCE_Loss", "min")
    
    if best_result is not None:
        print("\n" + "="*60)
        print("BEST HYPERPARAMETERS:")
        print("="*60)
        for param, value in best_result.config.items():
            print(f"{param:15}: {value}")
        
        print(f"\nBEST VALIDATION BCE LOSS: {best_result.metrics['BCE_Loss']:.6f}")
        print("="*60)
        
        return best_result.config, best_result.metrics['BCE_Loss']
    else:
        print("ERROR: No best trial found!")
        return None, None


if __name__ == "__main__":
    best_config, best_loss = main()