# %%
import os

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.preprocessing.data_registry import PATH, ENERCON_IDS, HIGH_RES_IDS
from data.preprocessing.sensors.enercon import FEATURES

def plot_all_feature_ranges(df_ranges, features, save_dir="influx_sensor_ranges"):
    """Function for plotting min max ranges for each sensor and energy system"""
    os.makedirs(save_dir, exist_ok=True)
    
    skip_feature = 'Energiezähler_der_Windenergieanlage_in_kWh_(Onlinewert)'
    
    features_to_plot = [f for f in features if f != skip_feature]
    
    print(f"Creating plots for {len(features_to_plot)} features...")
    
    for feature in features_to_plot:
        min_row = f"{feature}_min"
        max_row = f"{feature}_max"
        
        es_ids = df_ranges.columns
        mins = df_ranges.loc[min_row]
        maxs = df_ranges.loc[max_row]
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(range(len(es_ids)), (mins + maxs) / 2, 
                    yerr=[(maxs - mins) / 2, (maxs - mins) / 2], 
                    fmt='o', capsize=5)
        
        plt.xticks(range(len(es_ids)), es_ids, rotation=45)
        plt.title(f'{feature} ranges across sensors')
        plt.ylabel(feature)
        plt.tight_layout()
        
        clean_feature_name = feature.replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('*', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        filename = f"{clean_feature_name}_ranges.png"
        filepath = os.path.join(save_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")
    
    print(f"\nAll plots saved to '{save_dir}' directory!")


def get_sensor_ranges_powerdata():
    ids = ENERCON_IDS
    data_dict = {}

    for es_id in ids:
        path_sensor = PATH / "sensors_2000" / f"{es_id}_raw.parquet"
        df_sensors = pd.read_parquet(path_sensor)
        column_data = []
        for feature in ["power", "wind_speed", "nrot", "gondel_pos", "meter_reading"]:
            min = df_sensors[feature].min()
            max = df_sensors[feature].max()
            column_data.extend([min, max])
        data_dict[es_id] = column_data
    index = []
    for feature in ["power", "wind_speed", "nrot", "gondel_pos", "meter_reading"]:
        index.extend([f"{feature}_min", f"{feature}_max"])
    df_ranges = pd.DataFrame(data_dict, index = index)
    return df_ranges


def get_sensor_ranges_influx():
    ids = HIGH_RES_IDS
    data_dict = {}
    for es_id in ids:
        path_sensor = PATH / "sensors" / f"{es_id}"
        dfs = []
        for year in [2022, 2023, 2024]:
            df_sensors = pd.read_parquet(path_sensor / f"{year}_preprocessed.parquet")
            dfs.append(df_sensors)
        df_sensors = pd.concat(dfs)
        
        column_data = []
        for feature in FEATURES:
            min = df_sensors[feature].min()
            max = df_sensors[feature].max()
            column_data.extend([min, max])
        data_dict[es_id] = column_data

    index = []
    for feature in FEATURES:
        index.extend([f"{feature}_min", f"{feature}_max"])
    df_ranges = pd.DataFrame(data_dict, index = index)
    return df_ranges


def get_training_stats():
    df = pd.read_csv(PATH / "training_summary.csv")
    return df


def plot_model_performance_differences(df1, df2, title="Performance Differences Between Datasets"):
    """Plot differences between model performances between the two datasets (across horizons);
    two plots one for precision one for recall
    """
    
    common_horizons = df1.index.intersection(df2.index)
    
    df1_common = df1.loc[common_horizons]
    df2_common = df2.loc[common_horizons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    model_styles = {
        'XGBoost': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        'TCN One-Hot': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
        'TCN K-means': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'}
    }
    
    # Plot Precision Differences (left subplot)
    precision_columns = ['precision_xgb', 'precision_onehot', 'precision_kmeans']
    for i, col in enumerate(precision_columns):
        model_name = list(model_styles.keys())[i]
        style = model_styles[model_name]
        
        # Calculate difference
        precision_diff = df1_common[col] - df2_common[col]
        
        ax1.plot(common_horizons, precision_diff, 
                color=style['color'], 
                marker=style['marker'], 
                linestyle=style['linestyle'],
                linewidth=2.5,
                markersize=8,
                label=model_name,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=style['color'])
    
    # Add horizontal line at y=0 for reference
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Customize precision difference plot
    ax1.set_xlabel('Horizon (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision Difference', fontsize=12, fontweight='bold')
    ax1.set_title('Precision Differences Across Time Horizons', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xticks(common_horizons)
    
    # Plot Recall Differences (right subplot)
    recall_columns = ['recall_xgb', 'recall_onehot', 'recall_kmeans']
    for i, col in enumerate(recall_columns):
        model_name = list(model_styles.keys())[i]
        style = model_styles[model_name]
        
        # Calculate difference
        recall_diff = df1_common[col] - df2_common[col]
        
        ax2.plot(common_horizons, recall_diff, 
                color=style['color'], 
                marker=style['marker'], 
                linestyle=style['linestyle'],
                linewidth=2.5,
                markersize=8,
                label=model_name,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=style['color'])
    
    # Add horizontal line at y=0 for reference
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Customize recall difference plot
    ax2.set_xlabel('Horizon (minutes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recall Difference', fontsize=12, fontweight='bold')
    ax2.set_title('Recall Differences Across Time Horizons', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xticks(common_horizons)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Add overarching title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    return fig


def get_standard_results():

    data = {
        'precision_xgb': [0.54, 0.48, 0.48, 0.45, 0.44],
        'recall_xgb': [0.59, 0.50, 0.46, 0.45, 0.44],
        'precision_onehot': [0.5573, 0.4299, 0.4367, 0.4259, 0.4621],
        'recall_onehot': [0.5415, 0.3684, 0.3246, 0.3193, 0.2902],
        'precision_kmeans': [0.5881, 0.4224, 0.3899, 0.4227, 0.3774],
        'recall_kmeans': [0.5209, 0.3755, 0.3380, 0.3115, 0.3315]
    }

    horizons = [10, 20, 60, 90, 120]

    df = pd.DataFrame(data, index=horizons)
    df.index.name = 'Horizon'
    return df


def get_highres_results():


    data = {
        'precision_xgb': [0.30, 0.28, 0.27, 0.26, 0.26, 0.21],
        'recall_xgb': [0.43, 0.42, 0.40, 0.39, 0.38, 0.32],
        'precision_onehot': [0.3097, 0.2729, 0.2842, 0.2098, 0.2263, 0.2235],
        'recall_onehot': [0.4824, 0.5678, 0.4171, 0.3889, 0.3131, 0.3838],
        'precision_kmeans': [0.3094, 0.3571, 0.2869, 0.2633, 0.2486, 0.2619],
        'recall_kmeans': [0.4975, 0.3518, 0.3417, 0.4747, 0.4444, 0.1667]
    }

    horizons = [1, 10, 20, 60, 90, 120]

    df_highres = pd.DataFrame(data, index=horizons)
    df_highres.index.name = 'Horizon'

    return df_highres


def plot_model_performance(df, title="Model Performance Analysis"):
    """Plot Model performance for recall and precision"""
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    

    model_styles = {
        'XGBoost': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        'TCN One-Hot': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},
        'TCN K-means': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'}
    }
    

    precision_columns = ['precision_xgb', 'precision_onehot', 'precision_kmeans']
    for i, col in enumerate(precision_columns):
        model_name = list(model_styles.keys())[i]
        style = model_styles[model_name]
        
        ax1.plot(df.index, df[col], 
                color=style['color'], 
                marker=style['marker'], 
                linestyle=style['linestyle'],
                linewidth=2.5,
                markersize=8,
                label=model_name,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=style['color'])

    ax1.set_xlabel('Horizon (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Precision Performance Across Time Horizons', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xticks(df.index)
    ax1.set_ylim(0.2, 0.6)
    
    recall_columns = ['recall_xgb', 'recall_onehot', 'recall_kmeans']
    for i, col in enumerate(recall_columns):
        model_name = list(model_styles.keys())[i]
        style = model_styles[model_name]
        
        ax2.plot(df.index, df[col], 
                color=style['color'], 
                marker=style['marker'], 
                linestyle=style['linestyle'],
                linewidth=2.5,
                markersize=8,
                label=model_name,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=style['color'])
    
    ax2.set_xlabel('Horizon (minutes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_title('Recall Performance Across Time Horizons', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xticks(df.index)
    ax2.set_ylim(0.2, 0.6)
    
    plt.tight_layout()
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    return fig


def concat_tickets(long_hist):
    ts = []
    for id_ in tqdm(ENERCON_IDS):
        t = pd.read_parquet(PATH / f"tickets{"_2000" if long_hist else ""}" / f"{id_}_filtered.parquet")
        ts.append(t)
    ts = pd.concat(ts)
    return ts


def get_tickets_main_components(long_hist):
    ts = concat_tickets(long_hist)
    return ts["main_component"].value_counts()


def bar_plot_count_series(series):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=series.index,
        y=series.values,
        palette='viridis'
    )

    plt.xlabel('Main Components', fontsize=12)
    plt.ylabel('# Failures', fontsize=12)
    plt.xticks(rotation=70, ha='right', fontsize=10)

    for i, v in enumerate(series.values):
        ax.text(i, v + 0.1, str(v), ha='center', fontsize=9)

    plt.tight_layout()

    plt.show()


# %%
