import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from preprocessing.data_registry import PATH
########################################
# OFFSETS FOR LONG HIST:
# 0     10min
# 1     20min
# 5     60min
# 8     90min
# 11    2h
# OFFSETS FOR HIGH RES
# 0     1min
# 10    10min
# 19    20min
# 29    30min
# 59    60min
# 89    90min
# 119   2h
# %%
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

df = df.drop(columns=["es_id", "start_time", "end_time"], errors="ignore")

df = shuffle(df)

train_df = df[df["dataset"]=="train"].drop(columns=["dataset"])
val_df = df[df["dataset"]=="val"].drop(columns=["dataset"])

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_val = val_df.drop('label', axis=1)
y_val = val_df['label']

class_counts = y_train.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]  # negative/positive ratio

print(f"Class distribution in training data:")
print(f"Negative (0): {class_counts[0]}")
print(f"Positive (1): {class_counts[1]}")
print(f"Scale pos weight: {scale_pos_weight:.2f}")

#%% train
params = {'n_estimators': 466,
          'max_depth': 7,
          'learning_rate': 0.018152063249038626,
          'subsample': 0.861013101584684,
          'colsample_bytree': 0.7768953248067231,
          'min_child_weight': 9,
          'reg_alpha': 0.2313715587887067,
          'reg_lambda': 9.930616451197697e-05,
          'gamma': 0.16552801750337057}

highres_params = {'n_estimators': 199,
                  'max_depth': 10,
                  'learning_rate': 0.022841178523305628,
                  'subsample': 0.8356698028741715,
                  'colsample_bytree': 0.7247216208270185,
                  'min_child_weight': 6,
                  'reg_alpha': 0.0005795109436340742,
                  'reg_lambda': 0.006993007011557186, 
                  'gamma': 1.339940067248644e-07}
highres_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'scale_pos_weight': scale_pos_weight
})
params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'scale_pos_weight': scale_pos_weight
})
xgb_model = XGBClassifier(**params)
xgb_model.fit(X_train, y_train)

# %%Make predictions
y_train_pred = xgb_model.predict(X_train)
y_val_pred = xgb_model.predict(X_val)
y_train_proba = xgb_model.predict_proba(X_train)[:, 1]
y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
best_xgb = xgb_model
y_val_pred_best = y_val_pred
y_val_proba_best = y_val_proba
# %%Evaluate model performance
print("\n=== Training Set Performance ===")
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred))

print("\n=== Validation Set Performance ===")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

print("\nConfusion Matrix (Validation):")
print(confusion_matrix(y_val, y_val_pred))

# %%Feature importance
plt.figure(figsize=(12, 8))

feature_importance = best_xgb.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20 features

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('High Resolution Dataset: Top 20 XGB Features ')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
