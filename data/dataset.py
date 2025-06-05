# %%
# BaseEnerconReader reads data from persisted dataframes created in the preprocessing/main.py's script
# TimeSampleCreator is a class in which the functions are that draw positive and negative samlpes
# TabularFeatureBuilder has functions that transform samples from their time indexed dataformat into single rows of a dataframe
# dataframes for training XGB (tabular) were created with DataProcessor's new_baseline function/ the scripts at the end of this file.
# TensorBuilder is a class that has functions that uses samples and the persisted dataframes mentioned above to construct actual tensors used for training the TCN
# After experimenting with loading data for training from the full datapipeline, I found it more practical to persist the actual tensors. This is also convenient as
# negative samples tensors only needed to be persisted once as they are used across all training configurations
# DataProcessor is the main class that is initialized before loading data with "load_persisted_tensors".
#
#
#
#
#
#

import torch
import gc
import os
try:
    os.chdir("C:/Users/Paul.Wecker/dev/Studies/predictive_maintenance/")
except FileNotFoundError:
    print("Working on Cluster.")
from tqdm import tqdm
import random
from sklearn.utils import shuffle

from pathlib import Path
import pandas as pd
import numpy as np
from torch import cat, tensor, long, stack, float32
from sklearn.model_selection import train_test_split

from data.preprocessing.data_registry import ENERCON_IDS, HIGH_RES_IDS
try:
    from powerdata.data.model import CONFIG
    PATH = Path(CONFIG.data_repository_path / "predictive_maintenance" / "EnerconOpcXmlDaCs82a")
except:
    PATH = Path("../data")
    import ctypes

class BaseEnerconReader:
    """Base class for reading and preprocessing Enercon wind turbine data with caching."""
    
    def __init__(self, path=PATH, cluster=False, freq="1min", long_hist=False, kmeans=None):
        """Initialize the base reader with core configuration parameters.
        
        Args:
            path: Base path for data files (default: PATH)
            cluster: Whether running on a cluster (default: False)
            freq: Time frequency ('1min', '1h', or '10min') (default: '1min')
            long_hist: Whether to use long history data (default: False)
        """
        self.path = path
        self.cluster = cluster
        self.freq = freq
        self.long_hist = long_hist
        self.kmeans = kmeans
        # Validate frequency
        if self.freq not in ["1min", "1h", "10min"]:
            raise ValueError(f"time_resolution expects `1min`, `1h`, or `10min`, got: {self.freq}")
        
        # Override frequency if using long history
        if self.long_hist:
            self.freq = "10min"
            
        # Set up the data folder path
        if self.cluster:
            self.folder = Path("data" / f"training_dataframes{'_2000' if self.long_hist else ''}" / self.freq)
        else:
            self.folder = Path(self.path / f"training_dataframes{'_2000' if self.long_hist else ''}" / self.freq)
        
        # Cache for loaded data
        self._cache = {}


    def _get_cache_key(self, data_type, es_id):
        """Generate a unique cache key for a data type and energy system ID.
        
        Args:
            data_type: Type of data ('sensors', 'labels', or 'codes')
            es_id: Energy system ID
            
        Returns:
            String cache key
        """
        # Include long_hist in the key to ensure different versions don't clash
        history_suffix = "_long" if self.long_hist else "_short"
        return f"{data_type}_{es_id}{history_suffix}"


    def read_data(self, data_type, es_id):
        """Read data with caching to avoid redundant disk reads.
        
        Args:
            data_type: Type of data ('sensors', 'labels', or 'codes')
            es_id: Energy system ID
            
        Returns:
            DataFrame containing the requested data
        """
        cache_key = self._get_cache_key(data_type, es_id)
        
        # Check if data is already in cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Read data from disk
        if data_type == "codes":
            file_path = self.folder / f"{es_id}_{data_type}{f"_kmeans_{self.kmeans}" if self.kmeans is not None else ""}.parquet"
        else:
            file_path = self.folder / f"{es_id}_{data_type}.parquet"
        data = pd.read_parquet(file_path)
        
        # Cache the data
        self._cache[cache_key] = data
        
        return data


    def read_sensors(self, es_id: int) -> pd.DataFrame:
        """Read sensor data for a given energy system.
        
        Args:
            es_id: ID of the energy system
            
        Returns:
            DataFrame containing sensor data
        """
        return self.read_data("sensors", es_id)


    def read_labels(self, es_id: int) -> pd.DataFrame:
        """Read label data for a given energy system.
        
        Args:
            es_id: ID of the energy system
            
        Returns:
            DataFrame containing label data
        """
        return self.read_data("labels", es_id)


    def read_codes(self, es_id: int) -> pd.DataFrame:
        """Read error code data for a given energy system.
        
        Args:
            es_id: ID of the energy system
            
        Returns:
            DataFrame containing error code data
        """
        return self.read_data("codes", es_id)


    def clear_cache(self):
        """Clear the data cache to free memory."""
        self._cache = {}


    def filter_codes(self, df_codes, min_occurrences: int) -> pd.DataFrame:
        """Filter codes that appear less than the minimum number of occurrences.
        
        Args:
            df_codes: DataFrame containing error codes
            min_occurrences: Minimum number of occurrences for a code to be kept
            
        Returns:
            Filtered DataFrame with only frequently occurring codes
        """
        if self.cluster:
            folder = Path("data")
        else:
            folder = Path(self.path)

        df_stats = pd.read_csv(folder / f"training_dataframes{"_2000" if self.long_hist else ""}" / "code_counts.csv")
        keep_codes = df_stats[df_stats["count"] >= min_occurrences]["error_code"].unique()
        keep_codes = [str(code) for code in keep_codes if str(code) in df_codes.columns]

        return df_codes[keep_codes]


class TimeSampleCreator(BaseEnerconReader):
    """Creates time-based samples for predictive maintenance models."""
    
    def get_positive_samples(self, es_id, window_length, offset, freq, long_hist):
        labels = pd.read_parquet(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq / f"{es_id}_labels.parquet")
        labels["diff"] = labels["label"].diff()
        ticket_starts = labels[labels["diff"]==1].index
        windows = []
        freq = pd.tseries.frequencies.to_offset(freq)
        for start in ticket_starts:
            window_start = start - (window_length + offset) * freq
            window_end = start - (1 + offset) * freq
            windows.append([window_start, window_end])
        return windows


    def get_negative_samples(self, es_id, window_length, buffer, freq, long_hist, max_n):
        labels = pd.read_parquet(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq / f"{es_id}_all_tickets.parquet")
        windows = []
        freq = pd.tseries.frequencies.to_offset(freq)
        for i in range(0, len(labels), window_length):
            window_start = labels.index[i]
            window_end = window_start + freq * (window_length - 1)
            
            if window_end > labels.index[-1]:
                break
                
            buffer_end = window_end + freq * buffer
            buffer_end = min(buffer_end, labels.index[-1])

            window_ok = labels.loc[window_start:buffer_end]["label"].sum() == 0
            if window_ok:
                windows.append([window_start, window_end])

        num_windows = len(windows)
        if max_n > num_windows:
            return windows
        indices = random.sample(range(num_windows), max_n)
        windows_keep = [windows[i] for i in indices]
        return windows_keep


class TensorBuilder(BaseEnerconReader):
    """Builds tensor representations of time-series data for deep learning models."""


    def create_sensor_tensor(self, indices, es_id, window_length):
        df = self.read_sensors(es_id)
        tensors = []
        available_indices = []
        for i, x in enumerate(indices):
            start_time, end_time = x[0], x[1]
            slice_df = df.loc[start_time:end_time]
            if len(slice_df) == window_length:
                slice_tensor = tensor(slice_df.values, dtype=float32)
                tensors.append(slice_tensor)
                available_indices.append(i)
            else:
                if len(slice_df) > window_length // 2:
                    new_index = pd.date_range(start_time, end_time, freq=self.freq)
                    slice_df = slice_df.reindex(new_index).interpolate(method='time', limit_direction="both", limit=window_length // 2)
                    slice_tensor = tensor(slice_df.values, dtype=long)
                    tensors.append(slice_tensor)
                    available_indices.append(i)

        if len(tensors) == 0:
            return [], []
        return stack(tensors).permute(0, 2, 1), available_indices


    def create_codes_tensor(self, indices, es_id, window_length, filter = 0):
        """Create a tensor from error codes data.
        
        Args:
            indices: List of [start_time, end_time] lists
            es_id: Energy system ID
            
        Returns:
            PyTorch tensor containing codes data
        """
        df = self.read_codes(es_id)
        if filter:
            df = self.filter_codes(df, min_occurrences=filter)
        tensors = []
        num_codes = len(df.columns)
        for start_time, end_time in indices:
            slice_df = df.loc[start_time:end_time]
            if slice_df.empty:
                slice_tensor = tensor(np.zeros((window_length, num_codes)), dtype=long)
            elif len(slice_df) < window_length:
                data_tensor = tensor(slice_df.values, dtype=long)
                padding_tensor = tensor(np.zeros((window_length - len(slice_df), num_codes)), dtype=long)
                slice_tensor = cat([data_tensor, padding_tensor], dim=0)
            else:
                #try:
                    # Convert to numeric and handle non-numeric values
                 #   numeric_df = slice_df.apply(pd.to_numeric, errors='coerce').fillna(0)
                slice_tensor = tensor(slice_df.values, dtype=long)
                #except Exception as e:
                #    print(f"Error converting to tensor: {e}")
                #    print(slice_df)
                    # Provide a fallback - zeros or some appropriate default value
                #    slice_tensor = tensor(np.zeros((window_length, num_codes)), dtype=long)
            
            tensors.append(slice_tensor)

        return stack(tensors).permute(0, 2, 1)


    def create_label_tensor(self, y):
        """Create a tensor from labels.
        
        Args:
            y: List of labels (0s and 1s)
            
        Returns:
            PyTorch tensor containing labels
        """
        label_tensor = tensor(y, dtype=float32)
        return label_tensor


class TabularFeatureBuilder(BaseEnerconReader):
    """Creates tabular features for machine learning models like AutoGluon."""
    
    def summarize_code_timeseries(self, df_codes):
        """Create summary statistics for code timeseries.
        
        Args:
            df_codes: DataFrame containing error codes
            
        Returns:
            DataFrame with summary statistics
        """
        results = {}
        
        for code in df_codes.columns:
            code_series = df_codes[code]
            
            results[f"{code}_total_active_minutes"] = code_series.sum()
            
            state_changes = (code_series != code_series.shift()).sum()
            if len(code_series) > 0:
                state_changes = state_changes - 1 if state_changes > 0 else 0
            results[f"{code}_state_changes"] = state_changes

        
        result_df = pd.DataFrame([results])
        
        return result_df


    def summarize_sensor_timeseries(self, df_sensors):
        """Create summary statistics for sensor timeseries.
        
        Args:
            df_sensors: DataFrame containing sensor data
            
        Returns:
            DataFrame with summary statistics
        """
        results = {}
        
        for sensor in df_sensors.columns:
            sensor_series = df_sensors[sensor]

            results[f"{sensor}_min"] = sensor_series.min()
            results[f"{sensor}_max"] = sensor_series.max()
            results[f"{sensor}_mean"] = sensor_series.mean()
            results[f"{sensor}_std"] = sensor_series.std()
            
            time_diff = sensor_series.index.to_series().diff().dt.total_seconds() / 60  # Convert to minutes
            value_diff = sensor_series.diff().abs()
            rate_of_change = pd.DataFrame({'time_diff': time_diff, 'value_diff': value_diff}).dropna()
            if not rate_of_change.empty:
                avg_rate_per_minute = (rate_of_change['value_diff'] / rate_of_change['time_diff']).mean()
                results[f"{sensor}_avg_change_rate"] = avg_rate_per_minute
            else:
                results[f"{sensor}_avg_change_rate"] = float('nan')
        
        result_df = pd.DataFrame([results])
        
        return result_df


class DataProcessor(BaseEnerconReader):
    """Main interface for predictive maintenance data processing.
    
    This class centralizes configuration parameters and provides methods for data processing tasks.
    """
    
    def __init__(self, 
                 # Base parameters
                 ids=ENERCON_IDS,
                 long_hist=True,
                 freq="10min", 
                 cluster=False, 
                 path=PATH,
                 kmeans= 12,
                 buffer = 12,
                 offset = 0,
                 window_length=6*24 * 2,
                 horizon=6*24*7,
                 train_val_ratio=0.7,
                 min_code_occurrence=11,
                 step_size=1,
                 gap=0):
        # Initialize base class
        super().__init__(path, cluster, freq, long_hist, kmeans=kmeans)
        
        # Store additional parameters
        self.kmeans = kmeans
        self.offset = offset
        self.buffer = buffer
        self.ids = ids
        self.train_val_ratio = train_val_ratio
        self.default_window_length = window_length
        self.default_horizon = horizon
        self.default_min_code_occurrence = min_code_occurrence
        self.default_step_size = step_size
        self.default_gap = gap
        # Initialize specialized processors with same base configuration
        self.sample_creator = TimeSampleCreator(path, cluster, freq, long_hist, kmeans=kmeans)
        self.tensor_builder = TensorBuilder(path, cluster, freq, long_hist, kmeans=kmeans)
        self.tabular_builder = TabularFeatureBuilder(path, cluster, freq, long_hist, kmeans=kmeans)
        
        # Share the cache across all instances to avoid redundant reads
        self.sample_creator._cache = self._cache
        self.tensor_builder._cache = self._cache
        self.tabular_builder._cache = self._cache


    def cm_starts_with_labels_pos_neg(self, ids=None):
        if ids is None:
            ids = ENERCON_IDS
        else:
            ids = self.ids
        data_rows = []
        for id_ in ids:
            # x, y = self.sample_creator.condition_monitoring_samples(id_, self.default_window_length, self.default_horizon, self.default_step_size)
            x_pos = self.sample_creator.get_positive_samples(id_, self.default_window_length, self.offset, self.freq, self.long_hist)
            max_n = len(x_pos) * 2
            x_neg = self.sample_creator.get_negative_samples(id_, self.default_window_length, self.buffer, self.freq, self.long_hist, max_n)
            for i, (start_time, end_time) in enumerate(x_pos):
                data_rows.append({
                    'es_id': id_,
                    'start_time': start_time,
                    "end_time": end_time,
                    'label': 1
                })
            for i, (start_time, end_time) in enumerate(x_neg):
                data_rows.append({
                    'es_id': id_,
                    'start_time': start_time,
                    "end_time": end_time,
                    'label': 0
                })

        return pd.DataFrame(data_rows)
    

    def split_by_es_id_total_positives(self, df, train_size=0.7, label_col='label', es_id_col='es_id', 
                                       positive_value=1, random_state=42, max_attempts=100):
        
        es_stats = df.groupby(es_id_col).agg(
            total_records=pd.NamedAgg(column=label_col, aggfunc='count'),
            positive_count=pd.NamedAgg(column=label_col, aggfunc=lambda x: (x == positive_value).sum()),
            first_time=pd.NamedAgg(column='start_time', aggfunc='min')
        )
        
        es_stats['positive_pct'] = es_stats['positive_count'] / es_stats['total_records'] * 100
        es_stats = es_stats.sort_values('first_time')
        
        es_ids = es_stats.index.tolist()
        
        total_positive_samples = es_stats['positive_count'].sum()
        target_train_positives = int(total_positive_samples * train_size)
        
        best_split = None
        best_imbalance = float('inf')
        
        for _ in range(max_attempts): # loop for testing splits
            train_es_ids, val_es_ids = train_test_split(
                es_ids, train_size=train_size, random_state=random_state+_
            )
            
            train_positive = es_stats.loc[train_es_ids, 'positive_count'].sum()
            train_total = es_stats.loc[train_es_ids, 'total_records'].sum()
            val_positive = es_stats.loc[val_es_ids, 'positive_count'].sum()
            val_total = es_stats.loc[val_es_ids, 'total_records'].sum()
            
            if train_total == 0 or val_total == 0:
                continue
            
            train_positive_diff = abs(train_positive - target_train_positives)
            imbalance = train_positive_diff
            
            if imbalance < best_imbalance:
                best_imbalance = imbalance
                
                best_split = {
                    'train_es_ids': train_es_ids,
                    'val_es_ids': val_es_ids,
                    'train_records': train_total,
                    'val_records': val_total,
                    'train_positive_count': train_positive,
                    'val_positive_count': val_positive,
                    'target_train_positives': target_train_positives,
                    'train_size_pct': train_total / (train_total + val_total) * 100,
                    'positive_count_diff': train_positive_diff
                }
        
        if best_split is None:
            raise ValueError("Could not find a valid split")
        
        train_df = df[df[es_id_col].isin(best_split['train_es_ids'])]
        val_df = df[df[es_id_col].isin(best_split['val_es_ids'])]
        
        return {
            'train': train_df,
            'val': val_df,
            'metrics': best_split
        }


    def persist_neg_tensors(self):
        ids = self.ids
        df = pd.read_parquet(PATH / f"training_dataframes{"_2000" if self.long_hist else ""}" / f"negative_metadata_all.parquet")
        df = shuffle(df)
        for id_ in tqdm(ids):
            file_path = Path(PATH / f"training_dataframes{"_2000" if self.long_hist else ""}"
                                    / "tensors"
                                     / f"{id_}_all_negative_window_{self.default_window_length}_tensors{f"_kmeans_{self.kmeans}" if self.kmeans is not None else ""}.pt")
            if not file_path.exists():
                df_id = df[df["es_id"] == id_]
                x = []
                y = []
                for s, e, l in df_id[["start_time", "end_time", "label"]].values:
                    x.append([s, e])
                    y.append(l)

                sensors, available_indices = self.tensor_builder.create_sensor_tensor(x, id_, window_length=self.default_window_length)
                
                # handle empty indices (no sensor data found)
                if len(available_indices) == 0:
                    num_sensor_features = 5 if self.long_hist else 54 # normal features set vs shigh res
                    df_codes = self.tensor_builder.read_codes(id_)
                    if self.default_min_code_occurrence and self.kmeans is not None:
                        df_codes = self.tensor_builder.filter_codes(df_codes, min_occurrences=self.default_min_code_occurrence)
                    num_code_features = len(df_codes.columns) if not df_codes.empty else 1
                    empty_sensors = torch.empty(0, num_sensor_features, self.default_window_length, dtype=torch.float32)
                    empty_codes = torch.empty(0, num_code_features, self.default_window_length, dtype=torch.long)
                    empty_labels = torch.empty(0, dtype=torch.float32)
                    tensors = {
                        'sensors': empty_sensors,
                        'codes': empty_codes,
                        'labels': empty_labels,
                        "es_id": id_,
                        "dataset": df_id["dataset"].iloc[0] if not df_id.empty else "unknown"
                    }
                    torch.save(tensors, file_path)
                    print(f"Saved empty tensors for id {id_}")
                    continue

                x = [x[i] for i in available_indices]
                y = [y[i] for i in available_indices]
                codes = self.tensor_builder.create_codes_tensor(x,
                                                                id_,
                                                                self.default_window_length,
                                                                filter=0 if self.kmeans is None else self.default_min_code_occurrence)
                print(codes.shape)
                labels = self.tensor_builder.create_label_tensor(y)
                tensors = {
                    'sensors': sensors,
                    'codes': codes,
                    'labels': labels,
                    "es_id": id_,
                    "dataset": df_id["dataset"].loc[0]}
                torch.save(tensors, file_path)


    def persist_pos_tensors(self, ids = ENERCON_IDS):
        ids = self.ids
        offset = self.offset
        
        df = pd.read_parquet(PATH / f"training_dataframes{"_2000" if self.long_hist else ""}" / f"positive_metadata_offset_{offset}.parquet")
        df = shuffle(df)
        for id_ in tqdm(ids):
            file_path = Path(PATH / f"training_dataframes{"_2000" if self.long_hist else ""}"
                                    / "tensors"
                                     / f"pos_tensors_window_{self.default_window_length}_offset_{offset}{f"_kmeans_{self.kmeans}" if self.kmeans is not None else ""}"
                                     / f"{id_}_tensors.pt")
            if not file_path.exists():
                df_id = df[df["es_id"] == id_]
                x = []
                y = []
                for s, e, l in df_id[["start_time", "end_time", "label"]].values:
                    x.append([s, e])
                    y.append(l)

                sensors, available_indices = self.tensor_builder.create_sensor_tensor(x, id_, window_length=self.default_window_length)
                x = [x[i] for i in available_indices]
                y = [y[i] for i in available_indices]
                codes = self.tensor_builder.create_codes_tensor(x,
                                                                id_,
                                                                self.default_window_length,
                                                                filter=0 if self.kmeans is None else self.default_min_code_occurrence)
                print(codes.shape)
                labels = self.tensor_builder.create_label_tensor(y)
                tensors = {
                    'sensors': sensors,
                    'codes': codes,
                    'labels': labels,
                    "es_id": id_,
                    "dataset": df_id["dataset"].loc[0]}
                torch.save(tensors, file_path)


    def new_baseline(self):
        df_info = self.cm_starts_with_labels_pos_neg(self)
        split_metrics = self.split_by_es_id_total_positives(df_info, cm=True)
        train_ids = split_metrics["train"]["es_id"].unique()
        # val_ids = split_metrics["val"]["es_id"].unique()

        counter = 0 
        for id_ in tqdm(self.ids):
            if counter == 2:
                print("stopping program after 6 ids")
                break
            filename = f"{id_}_window_{self.default_window_length}_offset_{self.offset}_freq_{self.freq}.parquet"

            result_file = PATH / f"training_dataframes{'_2000' if self.long_hist else ''}" / "baselines" / filename
            if not result_file.exists():
                df_info_filtered = df_info[df_info["es_id"] == id_]
                dataset = "train" if id_ in train_ids else "val"
                df_sensors = self.read_sensors(id_)
                df_codes = self.read_codes(id_)
                
                if self.kmeans is None:
                    df_codes = self.filter_codes(df_codes, min_occurrences=self.default_min_code_occurrence)


                rows_list = []
                for s, e, label in df_info_filtered[["start_time", "end_time", "label"]].values:
                    df_codes_sample = df_codes[s:e]
                    df_sensors_sample = df_sensors[s:e]
                    series_codes_sample = self.tabular_builder.summarize_code_timeseries(df_codes_sample)
                    series_sensors_sample = self.tabular_builder.summarize_sensor_timeseries(df_sensors_sample.drop(columns=["es_id"], errors="ignore"))
                    
                    row_dict = {
                        'es_id': id_,
                        'start_time': s,
                        'end_time': e,
                        'label': label,
                        "dataset": dataset,
                        # "sample_id": sample_id
                    }
                    
                    for col in series_codes_sample.columns:
                        row_dict[col] = series_codes_sample[col].iloc[0]
                    
                    for col in series_sensors_sample.columns:
                        row_dict[col] = series_sensors_sample[col].iloc[0]
                    
                    rows_list.append(row_dict)
                df_samples = pd.DataFrame(rows_list)

                print("store df for id", id_)
                df_samples.to_parquet(result_file)
                if id_==self.ids[-1]:
                    print("finito")

                # counter += 1
                del df_sensors, df_codes, df_samples, rows_list
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except:
                    pass
            else:
                pass
        #     all_samples.append(df_samples)
        #     self._partial_cache_clear(id_)
            
        # df_final = pd.concat(all_samples)
        # return df_final


    def get_starts_with_labels(self, ids=None):
        """Create a DataFrame with columns:
            - es_id
            - start_time: the first time index/timestamp of the sample
            - label: the label of the sample
            This function uses create_all_samles to create a dataframe representing a full dataset, just without any features.
            It only has information on the start times, the labels and corresponding energysystem of the samples in the dataset. This function mainly serves as basis for experiments on how to split data, and for visualisations.
        """

        if ids is None:
            ids = self.ids

        data_rows = []
        for id_ in ids:
            x, y = self.sample_creator.create_all_samples(id_, self.default_window_length, self.default_horizon)
            for i, (start_time, _) in enumerate(x):
                data_rows.append({
                    'es_id': id_,
                    'start_time': start_time,
                    'label': y[i]
                })
    
        return pd.DataFrame(data_rows)


    def split_by_es_id(self, df, train_size=0.7, label_col='label', es_id_col='es_id', 
                    positive_value=1, random_state=42, max_attempts=100, cm=False):
        """
        Split data by es_id ensuring balanced label distribution
        """
        # Group by es_id and calculate statistics
        if cm:
            df["label"] = df["label"].apply(lambda x: 1 if x > 0 else 0)
        
        es_stats = df.groupby(es_id_col).agg(
            total_records=pd.NamedAgg(column=label_col, aggfunc='count'),
            positive_count=pd.NamedAgg(column=label_col, aggfunc=lambda x: (x == positive_value).sum()),
            first_time=pd.NamedAgg(column='start_time', aggfunc='min')
        )
        
        # Calculate positive label percentage for each es_id
        es_stats['positive_pct'] = es_stats['positive_count'] / es_stats['total_records'] * 100
        
        # Sort by first timestamp (optional, helps with reproducibility)
        es_stats = es_stats.sort_values('first_time')
        
        # Get unique es_ids
        es_ids = es_stats.index.tolist()
        
        # Track best split
        best_split = None
        best_imbalance = float('inf')
        
        # Try multiple random splits to find the most balanced one
        for _ in range(max_attempts):
            # Split the es_ids
            train_es_ids, val_es_ids = train_test_split(
                es_ids, train_size=train_size, random_state=random_state+_
            )
            
            # Calculate metrics for this split
            train_positive = es_stats.loc[train_es_ids, 'positive_count'].sum()
            train_total = es_stats.loc[train_es_ids, 'total_records'].sum()
            
            val_positive = es_stats.loc[val_es_ids, 'positive_count'].sum()
            val_total = es_stats.loc[val_es_ids, 'total_records'].sum()
            
            # Skip invalid splits
            if train_total == 0 or val_total == 0:
                continue
            
            train_pos_pct = train_positive / train_total * 100
            val_pos_pct = val_positive / val_total * 100
            
            # Calculate imbalance (difference in positive label percentages)
            imbalance = abs(train_pos_pct - val_pos_pct)
            
            # Check if this is the best split so far
            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best_split = {
                    'train_es_ids': train_es_ids,
                    'val_es_ids': val_es_ids,
                    'train_records': train_total,
                    'val_records': val_total,
                    'train_pos_pct': train_pos_pct,
                    'val_pos_pct': val_pos_pct,
                    'train_size_pct': train_total / (train_total + val_total) * 100
                }
        
        if best_split is None:
            raise ValueError("Could not find a valid split")
        
        # Create train and validation dataframes
        train_df = df[df[es_id_col].isin(best_split['train_es_ids'])]
        val_df = df[df[es_id_col].isin(best_split['val_es_ids'])]
        
        # Return the split dataframes and metrics
        return {
            'train': train_df,
            'val': val_df,
            'metrics': best_split
        }


    def nn_dataset(self, ids = ENERCON_IDS):
        """Prepare dataset for deep learning models.
        
        Args:
            ids: List of energy system IDs (default: None, uses self.ids)
            window_length: Length of the time window (default: None, uses self.default_window_length)
            horizon: Prediction horizon (default: None, uses self.default_horizon)
            
        Returns:
            Train and validation tensors for sensors, codes, and labels
        """
        # Use defaults if parameters not provided
        ids = self.ids
        window_length = self.default_window_length
        horizon = self.default_horizon
        gap = self.default_gap

        # Process each energy system
        train_sensors = []
        train_codes = []
        train_labels = []
        val_sensors = []
        val_codes = []
        val_labels = []
        
        # get df with es, starts and labels
        print("Compute Sample Starts/IDs/Labels")
        df = self.get_starts_with_labels()
        print("Find Data Splitt")
        split_metrics = self.split_by_es_id(df)

        train_ids = split_metrics["train"]["es_id"].unique()
        val_ids = split_metrics["val"]["es_id"].unique()
        print("Loading Tensors")
        for i, es_id in enumerate(ids):
            x, y = self.sample_creator.create_all_samples(es_id, window_length, horizon, gap=gap)
            
            sensors = self.tensor_builder.create_sensor_tensor(x, es_id)
            codes = self.tensor_builder.create_codes_tensor(x, es_id, window_length)
            labels = self.tensor_builder.create_label_tensor(y)
            
            if es_id in train_ids:
                train_sensors.append(sensors)
                train_codes.append(codes)
                train_labels.append(labels)
            elif es_id in val_ids:
                val_sensors.append(sensors)
                val_codes.append(codes)
                val_labels.append(labels)
            else:
                print("unknown id")

        
        return (cat(train_sensors, dim=0),
                cat(train_codes, dim=0),
                cat(train_labels, dim=0),
                cat(val_sensors, dim=0),
                cat(val_codes, dim=0),
                cat(val_labels, dim=0))


    def load_persisted_tensors(self):
        """Load 6 tensors for training TCNs: train/val sensor, code and label data"""
        offset = self.offset
        df_pos = pd.read_parquet(PATH / f"training_dataframes{"_2000" if self.long_hist else ""}" / f"positive_metadata_offset_{offset}.parquet")
        df_neg = pd.read_parquet(PATH / f"training_dataframes{"_2000" if self.long_hist else ""}" / "negative_metadata_all.parquet")
        df = pd.concat([df_pos, df_neg])
        train_ids = list(df[df["dataset"] == "train"]["es_id"].unique())
        val_ids = list(df[df["dataset"] == "val"]["es_id"].unique())
        train_sensors = []
        train_codes = []
        train_labels = []
        val_sensors = []
        val_codes = []
        val_labels = []

        for id_ in tqdm(self.ids):
            dataset = df[df["es_id"]==id_]["dataset"].iloc[0]
            pos_file_path = Path(PATH
                                / f"training_dataframes{"_2000" if self.long_hist else ""}"
                                / "tensors"
                                / f"pos_tensors_window_{self.default_window_length}_offset_{offset}{f"_kmeans_{self.kmeans}" if self.kmeans is not None else ""}"
                                / f"{id_}_tensors.pt")
            neg_file_path = Path(PATH
                                / f"training_dataframes{"_2000" if self.long_hist else ""}"
                                / "tensors"
                                / f"{id_}_all_negative_window_{self.default_window_length}_tensors{f"_kmeans_{self.kmeans}" if self.kmeans is not None else ""}.pt")
        
            pos_tensors = torch.load(pos_file_path) # set to true so torch shuts up
            neg_tensors = torch.load(neg_file_path, weights_only=False)

            if pos_tensors["es_id"] != id_:
                print("IDs do not match(positive)")
            if pos_tensors["dataset"] != dataset:
                print("Dataset assignments dont match(positive)")


            sensors_pos, sensors_neg = pos_tensors["sensors"], neg_tensors["sensors"]
            codes_pos, codes_neg = pos_tensors["codes"], neg_tensors["codes"]
            labels_pos, labels_neg = pos_tensors["labels"], neg_tensors["labels"]
            sensors = torch.cat([sensors_pos, sensors_neg], dim=0)
            codes = torch.cat([codes_pos, codes_neg], dim=0)
            labels = torch.cat([labels_pos, labels_neg], dim=0)
            num_samples = sensors.size(0)  # Get the length of dim=0
            perm_indices = torch.randperm(num_samples)

            # Apply the same permutation to all tensors
            sensors_shuffled = sensors[perm_indices]
            codes_shuffled = codes[perm_indices]
            labels_shuffled = labels[perm_indices]
            
            if id_ in train_ids:
                train_sensors.append(sensors_shuffled)
                train_codes.append(codes_shuffled)
                train_labels.append(labels_shuffled)
            elif id_ in val_ids:
                val_sensors.append(sensors_shuffled)
                val_codes.append(codes_shuffled)
                val_labels.append(labels_shuffled)
            else:
                print("ID could not be resolved.")
        return (cat(train_sensors, dim=0),
            cat(train_codes, dim=0),
            cat(train_labels, dim=0),
            cat(val_sensors, dim=0),
            cat(val_codes, dim=0),
            cat(val_labels, dim=0))


    def _partial_cache_clear(self, es_id):
        """Clear cache entries for a specific energy system to save memory.
        
        Args:
            es_id: Energy system ID to clear from cache
        """
        keys_to_remove = [k for k in self._cache if k.endswith(f"_{es_id}")]
        for key in keys_to_remove:
            del self._cache[key]


# %%
if __name__=="__main__":
    for offset in [0, 10, 19, 29, 59, 89, 119]:
        dp = DataProcessor(long_hist=False, # False
                           window_length=120, # 120
                           ids = HIGH_RES_IDS,
                           min_code_occurrence=0,
                           freq="1min",
                           kmeans=42,
                           offset=offset) # 59 89 119
        dp.persist_pos_tensors()
    dp.persist_neg_tensors()
    for offset in [0, 1, 5, 8, 11]:
        dp = DataProcessor(long_hist=True, # False
                        window_length=12, # 120
                        ids = ENERCON_IDS,
                        min_code_occurrence=0,
                        freq="10min",
                        kmeans=42,
                        offset=offset) # 59 89 119
        dp.persist_pos_tensors()
    dp.persist_neg_tensors()
    
    print("finito")
    # def persist_negative_stats():
        # df = pd.read_parquet(PATH / "training_dataframes" / "baselines" / "all_negative_samples.parquet")
        # df = df[["es_id", "dataset", "label", "start_time", "end_time"]]
        # df.to_parquet(PATH / "training_dataframes" / "negative_samples_stats.parquet")

    # def persist_positive_stats(offset, long_hist):
    #     if long_hist:
    #         freq = "10min"
    #         ids = ENERCON_IDS
    #         window = 12
    #     else:
    #         freq = "1min"
    #         ids = HIGH_RES_IDS
    #         window=120
    #     df = pd.read_parquet(PATH
    #                          / f"training_dataframes{"_2000" if long_hist else ""}"
    #                          / "baselines"
    #                          / f"final_window_{window}_offset_{offset}_freq_{freq}.parquet")
    #     df = df[["es_id", "dataset", "label", "start_time", "end_time"]]
    #     df.to_parquet(PATH
    #                     / f"training_dataframes{"_2000" if long_hist else ""}"
    #                     / f"positives_window_{window}_offset_{offset}.parquet")
    # long_hist = False
    # dfs = []
    # for id_ in HIGH_RES_IDS:
    #     df_neg = pd.read_parquet(PATH / "training_dataframes" / "baselines" / f"{id_}_negatives_new.parquet")
    #     dfs.append(df_neg)
    # if long_hist:
    #     freq = "10min"
    #     offset = 2 # change
    #     window = 6 * 2 # change
    # else:
    #     freq = "1min"
    #     offset = 2
    #     window = 60 * 24 * 2 # change
    # ids = ENERCON_IDS if long_hist else HIGH_RES_IDS
    # counts= pd.read_csv(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / "tickets_per_system.csv")
    # dp = DataProcessor(ids = ids, window_length=window, offset=offset, buffer = 60 * 24,  kmeans=None, freq=freq, min_code_occurrence=11, long_hist=False)
    # for id_ in tqdm(ids, desc="compute negatives"):
    #     result_file = Path(PATH / 
    #                        f"training_dataframes{"_2000" if long_hist else ""}" / 
    #                        "baselines" /
    #                        f"{id_}_negatives_window{window}.parquet")
    #     if result_file.exists():
    #         print(f"Skipping {id_}, file already exists")
    #         continue
    #     ticket_count  = counts[counts["es_id"]==id_]["label"].values[0]
    #     dataset = counts[counts["es_id"]==id_]["dataset"].values[0]

    #     print(f" Draw {9 * ticket_count} Samples.")
    #     windows = dp.sample_creator.get_negative_samples(id_, window_length=window, buffer = 60 * 24, freq=freq, long_hist=long_hist, max_n = 9 * ticket_count)
    #     df_sensors = dp.read_sensors(id_)
    #     df_codes = dp.read_codes(id_)
    #     rows_list = []
    #     for w in windows:
    #         s = w[0]
    #         e = w[1]
    #         df_codes_sample = df_codes[s:e]
    #         df_sensors_sample = df_sensors[s:e]
    #         if len(df_sensors_sample) < 120:
    #             if len(df_sensors_sample) > 50:
    #                 new_index = pd.date_range(s, e, freq=freq)
    #                 df_sensors_sample = df_sensors_sample.reindex(new_index).interpolate(method='time', limit_direction="both", limit=60)
    #             else:
    #                 continue

    #         series_codes_sample = dp.tabular_builder.summarize_code_timeseries(df_codes_sample)
    #         series_sensors_sample = dp.tabular_builder.summarize_sensor_timeseries(df_sensors_sample.drop(columns=["es_id"], errors="ignore"))
            
    #         row_dict = {
    #             'es_id': id_,
    #             'start_time': s,
    #             'end_time': e,
    #             'label': 0,
    #             "dataset": dataset,
    #             # "sample_id": sample_id
    #         }
            
    #         for col in series_codes_sample.columns:
    #             row_dict[col] = series_codes_sample[col].iloc[0]
            
    #         for col in series_sensors_sample.columns:
    #             row_dict[col] = series_sensors_sample[col].iloc[0]
            
    #         rows_list.append(row_dict)
    #     df_samples = pd.DataFrame(rows_list)
    #     print(f"Number of NaN values: ", sum(df_samples.isna().sum() > 0))
    #     print("store df for id", id_)
    #     print(f"Len of df: {len(df_samples)}")
    #     df_samples.to_parquet(result_file)
# %%
