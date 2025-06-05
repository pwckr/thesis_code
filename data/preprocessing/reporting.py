# %%
import os

import pandas as pd
import matplotlib.pyplot as plt

from data_registry import PATH, ENERCON_IDS, HIGH_RES_IDS
from sensors.enercon import FEATURES

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
