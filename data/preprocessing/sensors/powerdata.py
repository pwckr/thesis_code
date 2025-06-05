from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from powerdata.data.model import create_session, EnergySystem, Select, Record, CET
from data_registry import PATH, ENERCON_IDS

START = pd.Timestamp(2000, 1, 1, tzinfo=CET)
END = pd.Timestamp(2024, 12, 31, tzinfo=CET)


def __get_pyafm_data(start: pd.Timestamp, end: pd.Timestamp, es_id) -> pd.DataFrame:
    query = (
        Select(Record)
        .join(EnergySystem)
        .where(EnergySystem.id == es_id)
        .where(Record.timestamp >= start)
        .where(Record.timestamp <= end)
    )
    with create_session() as s:
        records = s.scalars(query).all()
        df = pd.DataFrame.from_records([
            {
                "timestamp":r.timestamp,
                "power":r.power,
                "wind_speed": r.wind_speed,
                "meter_reading": r.meter,
                "nrot": r.nrot,
                "ngen": r.ngen,
                "gondel_pos": r.gondel_pos,
                "es_id": es_id

            } for r in records
        ])
    
    return df
def filter_sensors(df):
    df = df[df["wind_speed"] < 50] # filter unrealistic scenarios (or heavy storms)
    df = df[df["nrot"] < 30] # filter away high records shown in data of four systems (we think its a sensor error)
    return df
def __preprocess_raw_pyafm_data(df: pd.DataFrame, es_id) -> pd.DataFrame:
    df = df.drop(columns=["temperature"], errors="ignore")
    df = df.set_index("timestamp")
    df = df.sort_index()
    df["meter_reading"] = df["meter_reading"].diff().shift(1)
    df = df.dropna()

    with create_session() as s:
        es = EnergySystem.by_id(s, es_id)
        nominal_power = es.type.nominal_power

    def average_angles(angles):
        angles = angles.to_numpy()
        if len(angles) == 0:
            return None

        angles = np.radians(angles)
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        return np.degrees(np.arctan2(sin_sum, cos_sum)) % 360  # get rid of negative degrees

    df = df.resample(rule="10min").agg(
        {"power": "mean",
         "wind_speed": "mean",
         "meter_reading": "sum",
         "nrot": "mean",
         "gondel_pos": average_angles,
         "es_id": "first"})
    
    angles = df['gondel_pos']
    nan_mask = angles.isna()
    
    if nan_mask.any():
        valid_angles = angles[~nan_mask]
        valid_indices = np.where(~nan_mask)[0]
        
        if len(valid_angles) >= 2:  # Need at least 2 points for interpolation
            # Convert valid angles to complex representation
            angles_rad = np.radians(valid_angles)
            valid_complex = np.exp(1j * angles_rad)
            
            # Create separate Series for real and imaginary parts
            real_series = pd.Series(index=df.index)
            imag_series = pd.Series(index=df.index)
            
            # Assign values only at valid indices
            real_series.iloc[valid_indices] = np.real(valid_complex)
            imag_series.iloc[valid_indices] = np.imag(valid_complex)
            
            # Interpolate both components with time method and limit=6 * 24
            real_interp = real_series.interpolate(method='time', limit=6*24)
            imag_interp = imag_series.interpolate(method='time', limit=6*24)
            
            # Convert back to angles where values were NaN
            complex_interp = real_interp + 1j * imag_interp
            angles_interp = np.degrees(np.angle(complex_interp)) % 360
            
            # Update only the NaN values
            df.loc[nan_mask, 'gondel_pos'] = angles_interp[nan_mask]
    
    # Interpolate all other numeric columns with limit=6
    numeric_cols = ['power', 'wind_speed', 'meter_reading', 'nrot']
    df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit=6 * 24)
    
    # Apply filter conditions after interpolation
    df = df[(df["meter_reading"] < nominal_power*1.2) & (df["meter_reading"] > 0)]
    df = df[['power', 'wind_speed', 'meter_reading', 'nrot', 'gondel_pos']]

    return df


def load_and_store_standard_sensors(start=START, end=END, ids=ENERCON_IDS) -> None:
    path = Path(PATH / "sensors_2000")
    for id_ in tqdm(ids, desc="Load & store PYAFM Data"):
        file_path = Path(path / f"{id_}_raw.parquet")
        if not file_path.exists():
            print(f"Loading PYAFM Sensors for: {id_}.")
            df = __get_pyafm_data(start, end, id_)
            if len(df):
                df.to_parquet(file_path)
                print(f"Added PYAFM-Sensors for {id_}.")
            else:
                print(f"No Sensor-Data for {id_}.")
        else:
            print(f"Sensors for {id_} found.")


def preprocess_and_store_standard_sensors(ids=ENERCON_IDS, overwrite=False) -> None:
    path = Path(PATH / "sensors_2000")
    for id_ in tqdm(ids, desc="Preprocess Standard Sensors"):
        file_path = Path(path / f"{id_}_raw.parquet")
        file_path_preprocessed = Path(path / f"{id_}_preprocessed.parquet")
        if (not file_path_preprocessed.exists()) or overwrite:
            if file_path.exists():
                print(f"Read raw sensors for: {id_}.")
                df = pd.read_parquet(file_path)

                if len(df):
                    df = __preprocess_raw_pyafm_data(df, id_)
                    df.to_parquet(file_path_preprocessed)
                    print(f"Stored Preprocessed Sensors for id {id_}")
                else:
                    print(f"Empty Sensor-Data for {id_}.")
            else:
                print(f"Raw sensors for {id_} NOT found.")
        else:
            print(f"Found preprocessed sensors for {id_}")