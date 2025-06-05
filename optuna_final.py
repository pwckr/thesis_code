# %%
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler

from data.preprocessing.data_registry import PATH

# data loading code
offset = 0
long_hist = False
if long_hist:
    freq = "10min"
    window = 12
else:
    freq = "1min"
    window = 120

df_pos = pd.read_parquet(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / "baselines" / f"final_window_{window}_offset_{offset}_freq_{freq}.parquet")
df_neg = pd.read_parquet(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / "baselines" / "all_negative_samples.parquet")

df = pd.concat([df_pos, df_neg], join="inner")
df = df.drop(columns=["es_id", "start_time", "end_time"])
df = shuffle(df)

train_df = df[df["dataset"]=="train"].drop(columns=["dataset"])
val_df = df[df["dataset"]=="val"].drop(columns=["dataset"])

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_val = val_df.drop('label', axis=1)
y_val = val_df['label']

class_counts = y_train.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]

class_weights = {0: 1.0, 1: scale_pos_weight}
sample_weights = y_train.map(class_weights)

# Create results file
results_filename = f'results_freq_{freq}_window_{window}_offset_{offset}.txt'
with open(results_filename, 'w') as f:
    f.write(f"=== XGBoost Bayesian Optimization Results (Validation Set - Weighted Log Loss) ===\n")
    f.write(f"Data configuration: freq={freq}, window={window}, offset={offset}\n\n")
    f.write(f"Class distribution in training data:\n")
    f.write(f"Negative (0): {class_counts[0]}\n")
    f.write(f"Positive (1): {class_counts[1]}\n")
    f.write(f"Scale pos weight: {scale_pos_weight:.2f}\n\n")
    f.write(f"Validation set size: {len(val_df)}\n")
    f.write(f"Training set size: {len(train_df)}\n\n")

print(f"Results will be saved to: {results_filename}")

def objective(trial):
    """Objective function for Optuna optimization using validation set and weighted log loss"""
    
    # hyperparameter search space
    params = { # these stay as they are
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight,
        
        # optimize these
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }
    # model init + training + retrieve val loss
    model = XGBClassifier(**params)
    
    model.fit(X_train, y_train)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    val_sample_weights = y_val.map(class_weights)
    return -log_loss(y_val, y_val_proba, sample_weight=val_sample_weights) # negative as optuna maximizes
    

# create optuna study and optimize
with open(results_filename, 'a') as f:
    f.write("=== Starting Bayesian Optimization (Using Validation Set - Weighted Log Loss) ===\n")
    
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)


n_trials = 100
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
# %% store results
with open(results_filename, 'a') as f:
    f.write(f"\n=== Optimization Results ===\n")
    f.write(f"Number of finished trials: {len(study.trials)}\n")
    f.write(f"Best trial:\n")
    f.write(f" Value (negative weighted log loss on validation): {study.best_value:.4f}\n")
    f.write(f"  Params: {study.best_params}\n\n")
