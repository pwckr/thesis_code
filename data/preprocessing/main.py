# %%
import os
os.chdir("C:/Users/Paul.Wecker/dev/Studies/predictive_maintenance/data/preprocessing")
from pathlib import Path
from typing import Tuple
import shutil

import pandas as pd
from  tqdm import tqdm
import numpy as np
import json

from codes import load_and_store_codes, create_bert_embeddings, preprocess_and_store_one_hot_codes, get_status, create_kmeans_assignments, merge_code_columns_by_cluster
from tickets import load_and_store_tickets, preprocess_and_store_binary_labels, filter_and_store_tickets, preprocess_and_store_all_tickets_into_binary
from sensors.enercon import filter_erronous_columns, resample_enercon, load_and_store_influx_sensors, preprocess_and_store_influx_sensors
from sensors.powerdata import filter_sensors, preprocess_and_store_standard_sensors
from data_registry import ENERCON_IDS, HIGH_RES_IDS
from powerdata.data.model import CET, CONFIG

PATH = Path(CONFIG.data_repository_path / "predictive_maintenance" / "EnerconOpcXmlDaCs82a")

# %%
def load_embedding_dict(model="bert"):
    # Später laden
    loaded_embeddings = np.load(Path(PATH / f'{model}_embeddings.npy'))
    with open(Path(PATH / 'codes.json'), 'r') as f:
        loaded_codes = json.load(f)

    loaded_dict = {code: embedding for code, embedding in zip(loaded_codes, loaded_embeddings)}
    return loaded_dict


def store_bert_embeddings_with_codes():
    codes, embeddings = create_bert_embeddings()
    np.save(Path(PATH /'bert_embeddings.npy', embeddings))
    with open(Path(PATH / 'codes.json'), 'w') as f:
        json.dump(codes, f)



def create_influx_training_df_with_freq(es_id: int, freq: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a time-horizon-naive dataset built from operational, scada and label data.

    This function assumes that:
        - the energy system is an "EnerconOpcXmlDaCs82a" system
        - that it has available influx data for years 2022, 2023 and 2024

    Arguments:
        es_id identifies energysystem

    Returns
        Three pd.DataFrames with one common time-index with freq:
            - one for sensor data from influx
            - one for scada codes
            - one for the labels (labels are time horizon naive)
    """

    path_sensor = PATH / "sensors" / f"{es_id}"
    path_scada = PATH / "codes" / f"{es_id}_{freq}_encoding.parquet"
    path_label = PATH / "labels" / "1min_binary" / f"{es_id}.parquet"

    try:
        dfs = []
        for year in [2022, 2023, 2024]:
            df_sensors = pd.read_parquet(path_sensor / f"{year}_preprocessed.parquet")
            df_sensors = filter_erronous_columns(df_sensors)
            dfs.append(df_sensors)
        df_sensors = pd.concat(dfs)
        df_codes = pd.read_parquet(path_scada)
        df_labels = pd.read_parquet(path_label)

        if freq != "1min":
            df_sensors = resample_enercon(df_sensors, freq=freq)
            df_codes = df_codes.resample(freq).max().fillna(0)
            df_labels = df_labels.resample(freq).max().fillna(0)
        df_sensors = df_sensors.dropna()

    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sensor_cols = df_sensors.columns
    codes_cols = df_codes.columns
    df_x = pd.merge(df_sensors, df_codes, left_index=True, right_index=True)
    new_index = df_x.index
    df_sensors = df_x[sensor_cols]
    df_codes = df_x[codes_cols]
    df_labels = df_labels.reindex(new_index , fill_value=0)
    return df_sensors, df_codes, df_labels


def resample_influx_training_dataframes(ids = ENERCON_IDS, freq="1h"):
    for id_ in tqdm(ids, desc= "Resampling"):
        sensors_source = Path(PATH / "training_dataframes" / "1min" /f"{id_}_sensors.parquet")
        labels_source = Path(PATH / "training_dataframes" / "1min" / f"{id_}_labels.parquet")
        codes_source = Path(PATH / "training_dataframes" / "1min" / f"{id_}_codes.parquet")
        df_sensors = pd.read_parquet(sensors_source)
        df_codes = pd.read_parquet(codes_source)
        df_labels = pd.read_parquet(labels_source)
        print(f"Resampling ID: {id_}")

        print(f"Resampling to freq: {freq}")
        sensor_file = Path(PATH / "training_dataframes" / freq /f"{id_}_sensors.parquet")
        codes_file = Path(PATH / "training_dataframes" / freq / f"{id_}_codes.parquet")
        labels_file = Path(PATH / "training_dataframes" / freq / f"{id_}_labels.parquet")

        if sensor_file.exists() and codes_file.exists() and labels_file.exists():
            print(f"{id_} DataFrames found.")
            pass
        else:
            df_sensors = resample_enercon(df_sensors, freq)
            df_codes = df_codes.resample(freq).max()
            df_labels = df_labels.resample(freq).max()

            if len(df_sensors) and len(df_codes) and len(df_labels):
                df_sensors.to_parquet(sensor_file)
                df_codes.to_parquet(codes_file)
                df_labels.to_parquet(labels_file)
                print(f"{id_} Resampled DataFrames stored.")
            else:
                print(f"{id_} No Data.")


def create_standard_training_df(es_id):
    path_sensors = Path(PATH / "sensors_2000" / f"{es_id}_preprocessed.parquet")
    path_codes = PATH / "codes_2000" / f"{es_id}_10min_onehot.parquet"
    path_labels = PATH / "labels_2000" / "10min_binary" / f"{es_id}.parquet"

    df_sensors = pd.read_parquet(path_sensors)
    df_sensors = filter_sensors(df_sensors)
    df_codes = pd.read_parquet(path_codes)
    df_labels = pd.read_parquet(path_labels)

    sensor_cols = df_sensors.columns
    code_cols = df_codes.columns
    df_sensors = df_sensors.dropna()

    df_x = pd.merge(df_sensors, df_codes, left_index=True, right_index=True)
    new_index = df_x.index
    df_sensors = df_x[sensor_cols]
    df_codes = df_x[code_cols]
    df_labels = df_labels.reindex(new_index , fill_value=0)
    return df_sensors, df_codes, df_labels


def compute_training_dataframes(ids=ENERCON_IDS, freq="1min", long_hist=False):
    for id_ in tqdm(ids, desc="Compute training DataFrames"):

        path = PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq


        codes_file = Path(path / f"{id_}_codes.parquet")
        sensors_file = Path(path / f"{id_}_sensors.parquet")
        labels_file = Path(path / f"{id_}_labels.parquet")
        if labels_file.exists() and sensors_file.exists() and codes_file.exists():
            pass
        else:
            if long_hist:
                sensors, codes, labels = create_standard_training_df(id_)
            else:
                sensors, codes, labels = create_influx_training_df_with_freq(id_, freq)

            if len(sensors) == 0:
                print(f"ID: f{id_} No training data available.")
            sensors.to_parquet(sensors_file)
            codes.to_parquet(codes_file)
            labels.to_parquet(labels_file)


def check_training_frames_for_nans(ids=ENERCON_IDS, freq="1h"):
        path = PATH / "training_dataframes" / freq
        for id_ in tqdm(ids, desc="Checking for NaNs"):
            sensors_df = pd.read_parquet(Path(path / f"{id_}_sensors.parquet"))
            codes_df = pd.read_parquet(Path(path / f"{id_}_codes.parquet"))
            labels_df = pd.read_parquet(Path(path / f"{id_}_labels.parquet"))
            # Print how many nans in each frame (including ID)
            print(f"Turbine ID: {id_}")
            print(f"  sensors_df: {sensors_df.isna().sum().sum()} NaN values")
            print(f"  codes_df: {codes_df.isna().sum().sum()} NaN values")
            print(f"  labels_df: {labels_df.isna().sum().sum()} NaN values")


def resample_binary_labels(ids=ENERCON_IDS, freq="1h"):
    path = PATH / "labels" 
    for id_ in tqdm(ids, desc=f"Resampling labels to {freq}"):
        df_labels = pd.read_parquet(path / "1min_binary" / f"{id_}.parquet")
        df_resampled = df_labels.resample(freq).max()
        df_resampled.to_parquet(path / f"{freq}_binary" / f"{id_}.parquet")


def load_long_restraint_hist(ids=ENERCON_IDS):
    dfs=[]
    path = Path(PATH / "tickets_2000")
    for id_ in tqdm(ids, desc="Load filtered Restraints"):
        file_path = Path(path / f"{id_}_filtered.parquet")    
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)
    df = pd.concat(dfs)
    return df


def ticket_code_eda(es_id):
    df_codes = pd.read_parquet(PATH / "codes_2000" / f"{es_id}.parquet")
    df_tickets = pd.read_parquet(PATH / "tickets_2000" / f"{es_id}_filtered.parquet")
    df_tickets = df_tickets[df_tickets["state"]=="Defekt"]
    df_infos = get_status()
    df_codes = pd.merge(df_codes, df_infos, left_on="error_code", right_on="error_code")[["start", "end", "info", "register"]]

    month = pd.Timedelta(days=31)
    result_dict = {}
    for row in df_tickets.iterrows():
        series = row[1]
        start = series["start"]
        end = series["end"]
        component = series["main_component"] + "_" + series["component"] + "_" + series["detail"]
        codes = df_codes[(df_codes["start"] < start) & (df_codes["start"] > start - month)].copy()
        codes["restraint_id"] = row[0]
        codes["problem"] = component
        result_dict[str(row[0]) + "_" + component] = codes
    
    return result_dict


def move_preprocessed_sensors_to_dataframe_folder():
    for es_id in tqdm(ENERCON_IDS, desc="move sensors"):
        # Create source and destination paths
        source_path = PATH / "sensors_2000" / f"{es_id}_preprocessed.parquet"
        dest_path = PATH / "training_dataframes_2000" / f"{es_id}_sensors.parquet"
        
        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
        else:
            print(f"Warning: Source file {source_path} does not exist")


def store_codestats(ids=ENERCON_IDS, long_hist=True):
    # read in starts of labels of training dataframes
    # first index (start)

    path = Path(PATH / f"codes{"_2000" if long_hist else ""}")
    path_label = Path(PATH / f"training_dataframes{"_2000" if long_hist else ""}")
    combined_counts = None
    for id_ in tqdm(ids):
        df_codes = pd.read_parquet(path / f"{id_}.parquet")
        df_labels = pd.read_parquet(path_label / f"{"10min" if long_hist else "1min"}" / f"{id_}_labels.parquet")
        print(id_, len(df_labels))
        start_labels = df_labels.index[0]
        df_codes = df_codes[df_codes["start"] >= start_labels]
        counts = df_codes["error_code"].value_counts()
        if combined_counts is None:
            combined_counts = counts
        else:
            combined_counts = combined_counts.add(counts, fill_value=0)
    df_counts = pd.DataFrame(combined_counts).reset_index()
    df_counts["error_code"] = df_counts["error_code"].astype("str")
    df_counts.to_csv(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / "code_counts.csv")


def load_code_columns(ids=ENERCON_IDS, long_hist=True, freq="10min"):
    """Creates a list of all codes (numbers not infos) found in the code-dataframes for all enercon systems.
    Code DataFrames contain all codes starting from 2000 until end of 2024."""
    cols = []
    for es_id in ids:
        codes = pd.read_parquet(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq / f"{es_id}_codes.parquet")
        for col in codes.columns:
            if col not in cols:
                cols.append(str(col))
    return cols


def extend_code_dataframe_columns(df, all_codes):
    print(f"Original columns: {len(df.columns)}")
    df = df.astype("int32")
    df.columns = df.columns.astype(str)
    current_cols = list(df.columns)
    all_codes_str = [str(code) for code in all_codes]
    new_codes = [code for code in all_codes_str if code not in current_cols]
    print(f"New codes to add: {len(new_codes)}")
    print(f"Total after extension: {len(df.columns) + len(new_codes)}")
    if new_codes:
        # Create new columns with standard numpy int32 dtype (not pandas nullable Int32)
        df_new = pd.DataFrame(
            0, 
            columns=new_codes, 
            index=df.index, 
            dtype="int32"  # This creates standard numpy int32, not pandas Int32
        )
        
        # Concatenate and ensure final dtype is standard numpy int32
        df_extended = pd.concat([df, df_new], axis=1)
        df_extended = df_extended.astype("int32")  # Ensure all columns are numpy int32
        return df_extended
    else:
        return df


def extend_code_dataframes_by_missing_codes(ids=ENERCON_IDS, freq="1min", long_hist=False):
    code_stats = pd.read_csv(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / "code_counts.csv")
    all_codes = list(code_stats["error_code"].unique())
    path = Path(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq)
    for id_ in tqdm(ids, desc="extending code columns"):
        df_codes = pd.read_parquet(path / f"{id_}_codes.parquet")
        df_codes_extended = extend_code_dataframe_columns(df_codes, all_codes)
        df_codes_extended.to_parquet(path / f"{id_}_codes.parquet")


def reformat_code_columns(ids = ENERCON_IDS, long_hist=True, freq="10min", kmeans=None):
    for id_ in tqdm(ids, desc="formatting column types"):
        file_path = Path(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq / f"{id_}_codes{f"_kmeans_{kmeans}" if kmeans is not None else ""}.parquet")
        df_codes = pd.read_parquet(file_path)
        for col in df_codes.columns:
            print(type(col))
            df_codes[col] = df_codes[col].astype("int32")
        df_codes.to_parquet(file_path)



def preprocess_store_codes_to_clusters(ids=ENERCON_IDS, N=42, freq="10min", long_hist=True):
    assignments = pd.read_csv(PATH / "training_dataframes_2000" / "code_counts_info_cluster.csv")

    for id_ in tqdm(ids, desc="assigning codes to clusters"):
        df_codes = pd.read_parquet(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq / f"{id_}_codes.parquet")
        df_codes_clustered = merge_code_columns_by_cluster(df_codes, assignments, N)
        df_codes_clustered.to_parquet(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq / f"{id_}_codes_kmeans_{N}.parquet")


# %%
if __name__ == "__main__":
    preprocess_store_codes_to_clusters(ids = ENERCON_IDS, freq="10min", long_hist=True)
    preprocess_store_codes_to_clusters(ids = HIGH_RES_IDS, freq="1min", long_hist=False)
    # preprocess_store_codes_to_clusters(freq="1min", long_hist=False)

    # standard dataset
    #
    #
    # start, end = pd.Timestamp(2000, 1, 1).tz_localize(CET), pd.Timestamp(2024, 12, 31).tz_localize(CET)

    # sensors
    # load_and_store_standard_sensors(start, end)
    # preprocess_and_store_standard_sensors()

    # # codes
    # # load_and_store_codes(start, end)
    # preprocess_and_store_one_hot_codes(freq="10min", long_hist=True)

    # # tickets
    # load_and_store_tickets(start, end, long_hist=True)
    # filter_and_store_tickets(long_hist=True)
    # preprocess_and_store_binary_labels(freq="10min", long_hist=True)

    # # training dataframes
    # compute_training_dataframes(freq="10min", long_hist=True)
    # store_codestats(long_hist=True)
    # extend_code_dataframes_by_missing_codes(freq="10min", long_hist=True)
    # debug_dataframe_dtypes(long_hist=True, freq="10min")
    # preprocess_store_codes_to_clusters(N=12, freq="10min", long_hist=True)

    # # influx dataset
    # #
    # #
    # # start, end = pd.Timestamp(2022, 10, 1).tz_localize(CET), pd.Timestamp(2024, 12, 31).tz_localize(CET)

    # # sensors
    # # load_and_store_influx_sensors()
    # preprocess_and_store_influx_sensors(freq="1min", overwrite=True)
    # preprocess_and_store_influx_sensors(freq="1h", overwrite=True)
    # codes
    # load_and_store_codes(start, end)
    # preprocess_and_store_one_hot_codes(freq="1min", long_hist=False)
    # preprocess_and_store_one_hot_codes(freq="1h", long_hist=False)

    # # # tickets
    # # # load_and_store_tickets(start, end, long_hist=False)
    # filter_and_store_tickets(long_hist=False)
    # preprocess_and_store_binary_labels(freq="1min", long_hist=False)
    # resample_binary_labels(freq="1h")

    # training dataset
    # compute_training_dataframes(freq="1min", long_hist=False)
    # # resample_influx_training_dataframes(freq="1h") # NEXT 2
    # preprocess_and_store_all_tickets_into_binary(ids=ENERCON_IDS, freq="10min", long_hist=True)
    # preprocess_and_store_all_tickets_into_binary(ids= HIGH_RES_IDS, freq="1min", long_hist=False)
    # store_codestats(ids= HIGH_RES_IDS, long_hist=False)
    # extend_code_dataframes_by_missing_codes(ids = HIGH_RES_IDS, freq="1min", long_hist=False)
    # extend_code_dataframes_by_missing_codes(freq="1h", long_hist=False)
    # preprocess_store_codes_to_clusters(ids = HIGH_RES_IDS, N=12, freq="1min", long_hist=False)
    # preprocess_store_codes_to_clusters(N=48, freq="1min", long_hist=False)
    # preprocess_store_codes_to_clusters(N=12, freq="1h", long_hist=False)
    # preprocess_store_codes_to_clusters(N=48, freq="1h", long_hist=False)
    # reformat_code_columns(long_hist=False, kmeans=12, freq="1h")
