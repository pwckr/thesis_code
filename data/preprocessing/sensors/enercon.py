import warnings
from typing import Tuple, List
from itertools import product
import os
os.chdir("C:/Users/Paul.Wecker/dev/Studies/power_forecast/maintenance_acquisition/preprocessing/sensors/")

from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd

from data_registry import ENERCON_TYPE as TYPE
from data_registry import PATH, ENERCON_IDS

try:
    from influxdb_client.client.warnings import MissingPivotFunction
    from powerdata.data.model import EnergySystem, create_session, CONFIG, CET, Select
    from powerdata.data.influx import InfluxDB
except ImportError:
    print("Could not import powerdata and influx_client.")

warnings.simplefilter("ignore", MissingPivotFunction)

FEATURES = [
    'Temperatur_Sys_1._Leistungsschrank_7_in_°C',
    'Produktionsdauer_in_min',
    'Gondelposition_der_Windenergieanlage_in_°_(Onlinewert)',
    'Temperatur_Sys_1._Leistungsschrank_9_in_°C',
    'Produzierte_Energie_in_kWh',
    'Rotordrehzahl_der_Windenergieanlage_in_U/min_(Onlinewert)',
    'Spinnertemperatur_in_°C',
    'Leistungsfaktor_cos_Phi_an_der_Windenergieanlage_(Onlinewert)',
    'maximale_Leistung_in_kW',
    'Temperatur_Lager_vorn_in_°C',
    'Temperatur_Steuerschrank_in_°C',
    'Temperatur_Blattregelschrank_Blatt_C_in_°C',
    'Temperatur_Sys_1._Leistungsschrank_10_in_°C',
    'Umgebungstemperatur_in_°C',
    'Blindleistung_der_Windenergieanlage_in_kvar_(Onlinewert)',
    'Temperatur_Sys_1._Leistungsschrank_3_in_°C',
    'Temperatur_Gondelsteuerschrank_in_°C',
    'Umgebungstemperatur_Gondel_in_°C',
    'mittlere_Blindleistung_in_kvar',
    'Leistung_der_Windenergieanlage_in_kW_(Onlinewert)',
    'Temperatur_Kühlblech_Gleichrichter_1_in_°C',
    'Temperatur_Blattregelschrank_Blatt_A_in_°C',
    'Temperatur_Kühlblech_Blattverstellung_Blatt_B_in_°C',
    'Temperatur_Stator_2_in_°C',
    'Temperatur_Kühlblech_Blattverstellung_Blatt_A_in_°C',
    'mittlere_Windgeschwindigkeit_in_m/s',
    'Temperatur_Sys_1._Leistungsschrank_6_in_°C',
    'maximale_Windgeschwindigkeit_in_m/s',
    'Temperatur_Sys_1._Leistungsschrank_1_in_°C',
    'Energiezähler_der_Windenergieanlage_in_kWh_(Onlinewert)',
    'Transformatortemperatur_in_°C',
    'mittlere_Rotordrehzahl_in_U/min',
    'Temperatur_Stator_1_in_°C',
    'Turmtemperatur_in_°C',
    'Temperatur_Rotor_2_in_°C',
    'Gondeltemperatur_in_°C',
    'Betriebsstunden_der_Windenergieanlage_in_h_(Onlinewert)',
    'Temperatur_Sys_1._Leistungsschrank_4_in_°C',
    'mittlere_Leistung_in_kW',
    'Temperatur_Gleichrichterschrank_in_°C',
    'Temperatur_Sys_1._Leistungsschrank_8_in_°C',
    'Gondelposition_in_º',
    'Temperatur_Blattregelschrank_Blatt_B_in_°C',
    'Mittelwert_Blattwinkel_über_A,_B,_C_in_°',
    'Temperatur_Sys_1._Leistungsschrank_11_in_°C',
    'Windgeschwindigkeit_an_der_Windenergieanlage_in_m/s',
    'Temperatur_Kühlblech_Erregerstellerschrank_in_°C',
    'Windfahne_der_Windenergieanlage_in_°_(Differenz_zwischen_Windrichtung_und_Gondelposition)',
    'Temperatur_Kühlblech_Blattverstellung_Blatt_C_in_°C',
    'Temperatur_Sys_1._Leistungsschrank_2_in_°C',
    'Temperatur_Kühlblech_Gleichrichter_2_in_°C',
    'Temperatur_Rotor_1_in_°C',
    'Temperatur_Lager_hinten_in_°C',
    'Temperatur_Sys_1._Leistungsschrank_5_in_°C'
]


def __get_influx_data(start: pd.Timestamp, end: pd.Timestamp,
                    identifier: int, field_names: List[str],
                    measurement: str = TYPE) -> pd.DataFrame:
    """Create raw DataFrame with requested Influx-Fields for given time period / energy system.

    Arguments:
        start, end: Define the time period
        es_id: Identify energy system
        field_names: Stores all names of fields of interest
        measurement: name of Influx measurement

    Returns:
        pd.DataFrame with columns:
        `result`, `table`, `_start`, `_stop`, `_time`,
        `EnergySystemName`, `IdentNr`, `measurement` and all field names.
    """

    PROXY_URL = "http://proxy.enertrag.de:3128"
    client = InfluxDB(proxy_url=PROXY_URL, server_url=CONFIG.influx_uri, timeout=100_000_000)

    start, end = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    query = f"""
    from(bucket: "Enertrag Betrieb")
        |> range(start: {start}, stop: {end})
        |> filter(fn: (r) => r["_measurement"] == "windTurbineOperationData_{measurement}")
        |> filter(fn: (r) => r["IdentNr"] == "{identifier}")
        |> filter(fn: (r) => {" or ".join([f"r[\"_field\"] == \"{measurement}.{field}\"" for field in field_names])})
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> drop(columns: ["table", "_start", "_stop", "result", "EnergySystemName", "_measurement"])
    """

    return client.execute(query)


def __clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Set and sort index, remove redundant columns."""
    df.index = df["_time"]
    df.index.names = ["Timestamp"]
    df = df.sort_index()
    df = df.drop(columns=["result", "table", "_time", "IdentNr"])  # insert below columns and add kwarg `errors="ignore"`
    for col in ["_start", "_stop", "EnergySystemName", "_measurement"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def __shift_nan_columns(df: pd.DataFrame) -> pd.DataFrame:
    nan_mask = df.iloc[0].isna()
    df1 = df.loc[:, nan_mask]
    df2 = df.loc[:, ~nan_mask]

    df1 = df1.dropna()
    df2 = df2.dropna()

    return df1, df2


def __rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to their influx-field-description."""
    path = Path(PATH / f"{TYPE}.csv")
    df_influx = pd.read_parquet(path)
    df_influx["Desc_"] = df_influx["Desc"].apply(lambda x: x.replace(" ", "_"))
    df_influx["Influx_Field_Name"] = df_influx["Influx_Field_Name"].apply(lambda x: TYPE + "." + x)
    column_mapping = dict(zip(df_influx["Influx_Field_Name"], df_influx["Desc_"]))
    column_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=column_mapping)
    return df


def resample_enercon(df, freq="1min") -> pd.DataFrame:
    """
    Resample Enercon data to the specified frequency, handling angular columns appropriately.
    
    Args:
        df: Input DataFrame with time index
        freq: Resampling frequency (default: "1min")
        
    Returns:
        Resampled DataFrame with proper interpolation
    """
    def average_angles(angles):
        angles = angles.to_numpy()
        if len(angles) == 0:
            return None

        angles = np.radians(angles)
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        return np.degrees(np.arctan2(sin_sum, cos_sum)) % 360  # get rid of negative degrees
    
    angle_cols = ["Mittelwert_Blattwinkel_über_A,_B,_C_in_°", "Gondelposition_in_º"]
    agg_dict = {col: (average_angles if col in angle_cols else "mean") for col in df.columns}
    
    # First resample and aggregate
    df_r = df.resample(rule=freq).agg(agg_dict)
    if freq == "1min":
        limit = 60
    elif freq == "1h":
        limit = 1
    else:
        limit = 10
    # Custom interpolation for angular columns
    for col in angle_cols:
        if col in df_r.columns:
            # Extract angles that need interpolation
            angles = df_r[col]
            nan_mask = angles.isna()
            
            if nan_mask.any():  # Only process if there are NaNs
                # Create arrays for the valid values
                valid_angles = angles[~nan_mask]
                valid_indices = np.where(~nan_mask)[0]
                
                if len(valid_angles) >= 2:  # Need at least 2 points for interpolation
                    # Convert valid angles to complex representation
                    angles_rad = np.radians(valid_angles)
                    valid_complex = np.exp(1j * angles_rad)
                    
                    # Create separate Series for real and imaginary parts
                    real_series = pd.Series(index=df_r.index)
                    imag_series = pd.Series(index=df_r.index)
                    
                    # Assign values only at valid indices
                    real_series.iloc[valid_indices] = np.real(valid_complex)
                    imag_series.iloc[valid_indices] = np.imag(valid_complex)
                    
                    # Interpolate both components with time method and limit
                    real_interp = real_series.interpolate(method='time', limit=limit)
                    imag_interp = imag_series.interpolate(method='time', limit=limit)
                    
                    # Convert back to angles where values were NaN
                    complex_interp = real_interp + 1j * imag_interp
                    angles_interp = np.degrees(np.angle(complex_interp)) % 360
                    
                    # Update only the NaN values
                    df_r.loc[nan_mask, col] = angles_interp[nan_mask]
    
    # Interpolate all non-angular columns normally
    non_angle_cols = [col for col in df_r.columns if col not in angle_cols]
    if non_angle_cols:
        df_r[non_angle_cols] = df_r[non_angle_cols].interpolate(limit=limit, method="time")
    
    return df_r


def filter_erronous_columns(df):
    df = df.drop(columns=["Temperatur_Sys_1._Leistungsschrank_9_in_°C",
                          "Temperatur_Sys_1._Leistungsschrank_10_in_°C", 
                          "Temperatur_Sys_1._Leistungsschrank_11_in_°C",
                          "Blindleistung_der_Windenergieanlage_in_kvar_(Onlinewert)"])
    return df


def get_ids(measurement: str = TYPE) -> Tuple[list[int], list[int]]:
    """Lists all energy system ids of energy systems with given measurement.

    Arguments:
        measurement: the data-source-connection-type of which the ids should be fetched

    Returns
        List of energy system ids (integers)
    """

    PROXY_URL = "http://proxy.enertrag.de:3128"
    client = InfluxDB(proxy_url=PROXY_URL, server_url=CONFIG.influx_uri, timeout=1000_000)

    query = """
    from(bucket: "Enertrag Betrieb")
    |> range(start: -1h)
    |> keyValues(keyColumns: ["_measurement", "IdentNr"])
    |> group()
    """

    df_idents = client.execute(query)

    df_idents = df_idents[df_idents["_measurement"] == f"windTurbineOperationData_{measurement}"]
    idents = list(df_idents["IdentNr"].unique())
    idents = [int(ident) for ident in idents]

    with create_session() as s:
        query = Select(EnergySystem.id).where(EnergySystem.identifier.in_(idents))
        ids = s.scalars(query).all()

    return ids, idents


def __get_field_names() -> List[str]:
    """Returns list of names of influx-fields(columns)."""

    path = Path(PATH / f"{TYPE}.csv")
    df_influx = pd.read_parquet(path)
    indexes = [
        1,  # windspeed (m/s)
        2,  # wind-vane (°) (difference gondola-position/wind-direction)
        3,  # rotor speed (turns/minute)
        4,  # gondola-position online-value (°)
        5,  # operation hours
        6,  # counter online-value (kWh)
        7,  # Power online-value (kW)
        8,  # performance factor
        9,  # idle power
        10, 11, 12,  # Voltages in U12, U23, U31
        13, 14, 15,  # current entry (Stromeinspeisung) L1, L2, L3
        16,  # grid frequency
        17, 18,  # AVG/Max wind-speed (m/s)´
        20,  # AVG rotor speed
        23, 24,  # AVG/Max Power kW
        26,  # gondola-position (°)
        28,  # counter (kW)
        29,  # duration of production (min)
        30,  # AVG idle power
        37,  # averaged angle of blades A, B, C (°)
        54, 55, 56, 57, 58, 59, 60, 61, 62,  # temperatures of components
        66, 67,  # Temperature of Rotor 1/2
        68, 69,  # Temperature of Stator 1/2
        70,  # outside temperature gondola (°C)
        71, 72, 73, 74, 75, 76,  # temperatures of different things
        77,  # outside temperature (°C)
        78,  # temperature tower (°C)
        79,  # temperature of control-closet (Steuerschrank)
        80,  # temperature transformer
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91  # temperatures of Power Cabinet (Leistungsschrank)
    ]
    field_names = df_influx.loc[indexes]["Influx_Field_Name"].to_list()
    return field_names


def preprocess_and_store_influx_sensors(freq, ids=ENERCON_IDS, overwrite=False) -> None:
    """Proprocess raw data and store result from given list of ids.

    If no list passed, function checks for available raw data and processes it.

    Parameters:
        ids: list with ids

    Returns:
        None
    """
    if len(ids) == 0:
        ids = os.listdir(PATH / "sensors")
    for id_ in tqdm(ids):
        for year in [2022, 2023, 2024]:
            file_path = Path(PATH / "sensors" / f"{id_}" / f"{year}_raw.parquet")
            if file_path.exists():
                result_file_path = Path(PATH / "sensors" / f"{id_}" / f"{year}_preprocessed.parquet")
                if not result_file_path.exists() or overwrite:
                    df = pd.read_parquet(file_path)
                    if len(df):
                        df = __clean_columns(df)
                        df = __rename_columns(df)
                        df1, df2 = __shift_nan_columns(df)
                        df1 = resample_enercon(df1, freq)
                        df2 = resample_enercon(df2, freq)
                        df = pd.merge(df1, df2, left_index=True, right_index=True)
                        df = df[FEATURES]
                        df.to_parquet(result_file_path)
                        print(f"{id_}-{year} stored.")
                    else:
                        print(f"{id_}-{year} empty.")
                        df.to_parquet(result_file_path)
                else:
                    print(f"{id_}-{year} already stored.")
            else:
                print(f"No file: {file_path}")
                print(f"{id_}-{year} no data.")


def load_and_store_influx_sensors(field_names = [], ids = ENERCON_IDS, years = [2022, 2023, 2024], overwrite=False) -> None:
    if len(field_names) == 0:
        field_names = __get_field_names()

    if len(ids) == 0:
        ids, idents = get_ids()
    else:
        query = (
            Select(EnergySystem.identifier)
            .where(EnergySystem.id.in_(ids))
        )
        with create_session() as s:
            idents = s.scalars(query).all()

    ids_ident = dict(zip(ids, idents))
    for year, id_ in tqdm(list(product(years, ids))):
        path = Path(PATH / "sensors" / str(id_))
        file_path = Path(path / f"{year}_raw.parquet")

        start, end = pd.Timestamp(year, 1, 1).tz_localize(CET), pd.Timestamp(year, 12, 31).tz_localize(CET)

        if os.path.exists(path):
            if not os.path.exists(file_path) or overwrite:

                print(f"Loading {year} {id_}.")
                delta = pd.Timedelta(days=(end - start).days / 4)
                periods = [((start + i * delta), (start + (i + 1) * delta)) for i in range(0, 4)]
                dfs = [__get_influx_data(_start, _end, ids_ident[id_], field_names) for _start, _end in periods]
                df = pd.concat(dfs)

                if len(df):
                    df.to_parquet(file_path)
                    print(f"Added Data for {year} - {id_}.")
                else:
                    print(f"No Data for {year} - {id_}.")
                    df.to_parquet(file_path)
        else:
            os.makedirs(path)

            print(f"Loading {year} {id_}.")
            delta = pd.Timedelta(days=(end - start).days / 4)
            periods = [((start + i * delta), (start + (i + 1) * delta)) for i in range(0, 4)]
            dfs = [__get_influx_data(_start, _end, ids_ident[id_], field_names) for _start, _end in periods]
            df = pd.concat(dfs)

            if len(df):
                df.to_parquet(file_path)
                print(f"Added Data for {year} - {id_}.")
            else:
                print(f"No Data for {year} - {id_}.")
