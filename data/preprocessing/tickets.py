# %%
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import numpy as np
# os.chdir("C:/Users/Paul.Wecker/dev/Studies/predictive_maintenance/")
from data_registry import ENERCON_IDS, PATH

try:
    from powerdata.data.model import EnergySystem, create_session, CET
except ImportError:
    print("Could not load powerdata.")


def __get_ticket_data(start: pd.Timestamp, end: pd.Timestamp, es_id: int) -> pd.DataFrame:
    """Create DataFrame with restraints-data.

    Arguments:
        start: start of period of interest
        end: end of period of interest
        es_id: identifies energy system

    Returns:
        DataFrame with columns `id`, `start`, `end`, `es_id`, `power_reduction`,
        `main_component`, `component`, `state`, `cause` and `detail`.
    """
    with create_session() as s:
        es = EnergySystem.by_id(session=s, id_=es_id)
        restraints = (
            p for p in es.procedures
            if (
                (p.start is not None)
                and (p.end is not None)
                and (p.end > start)
                and (p.start < end)
                and (p.type_id != 2)
            )
        )

        df = pd.DataFrame([
            {
                "id": r.id,
                "start": r.start,
                "end": r.end,
                "es_id": es_id,
                "power_reduction": r.power_reduction,
                "cause": r.cause.name if r.cause else pd.NA,
                "main_component": r.main_component.name if r.main_component is not None else pd.NA,
                "component": r.component.name if r.component is not None else pd.NA,
                "detail": r.detail.name if r.detail is not None else pd.NA,
                "state": r.state.name if r.state is not None else pd.NA,
            }
            for r in restraints
        ])

    return df


def __create_binary_labels(df: pd.DataFrame, freq) -> pd.DataFrame:
    """Create binary labels from a filtered restraints-DataFrame with given freq.

    Arguments:
        df: with restraints and columns 'start', 'end'. It is assumed that the DataFrame has been filtered before creating labels.
        freq: defines the frequency with a string

    Returns:
        pd.DataFrame with pd.DateTimeIndex and column `label` where 1 means there is an active restraint.
    """

    if len(df)==0:
        raise Exception("DataFrame Empty")
    freq_offset = pd.tseries.frequencies.to_offset(freq)
    freq_seconds = pd.Timedelta(freq_offset).total_seconds()

    start_time = df["start"].min().floor("h")

    if not pd.isna(df["end"].max()):
        end_time = df["end"].max().ceil("h")
    else:
        end_time = df["start"].max().ceil("h")
    time_index = pd.date_range(start_time, end_time, freq=freq)
    index_numpy = np.array([ts.timestamp() for ts in time_index])

    ends = df["end"].copy()
    for i, end in enumerate(ends):
        if pd.isna(end):
            ends.iloc[i] = df["start"].iloc[i] + freq_offset

    starts_numpy = np.array([ts.timestamp() for ts in df["start"]])
    ends_numpy = np.array([ts.timestamp() for ts in df["end"]])

    result_array = np.zeros(len(time_index), dtype=np.int8)

    for start, end in tuple(zip(starts_numpy, ends_numpy)):
        active_mask = (start <= index_numpy) & (end + freq_seconds > index_numpy)
        for j in np.where(active_mask)[0]:
            result_array[j] = 1

    return pd.DataFrame(index=time_index, data={"label": result_array})

# 1
def load_and_store_tickets(start, end, ids=ENERCON_IDS, long_hist=False) -> None:
    path = PATH / f"tickets{"_2000" if long_hist else ""}"
    for id_ in tqdm(ids, desc="Load & store tickets"):
        file_path = Path(path / f"{id_}.parquet")

        if not file_path.exists():
            df = __get_ticket_data(start, end, id_)

            if len(df):
                df.to_parquet(file_path)
                print(f"Added Ticket-Data for {id_}.")
            else:
                print(f"No Ticket-Data for {id_}.")
        else:
            print(f"Tickets for {id_} found.")

# 2
def filter_and_store_tickets(ids=ENERCON_IDS, long_hist=False) -> None:
    states_of_interest = [
        "Defekt",
        "Fettmangel",
        "Gebrochen",
        "Gerissen",
        "Risse",
        "Temperatur zu hoch",
        "Verschlissen",
        "Ölleckage"
    ]
    for id_ in tqdm(ids, desc="Load & store Labels"):
        file_path = PATH / f"tickets{"_2000" if long_hist else ""}" / f"{id_}.parquet"
        result_file_path = PATH / f"tickets{"_2000" if long_hist else ""}" / f"{id_}_filtered.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df = df[(df["cause"]=="Energieanlage") & df["state"].isin(states_of_interest)]
            df.to_parquet(result_file_path)
            print(f"Stored filtered tickets for id {id_}")
        else:
            print(f"Could not find tickets for id {id_}")

# 3
def preprocess_and_store_all_tickets_into_binary(ids=ENERCON_IDS, freq="10min", long_hist = False):
    """Load and stores labels from given ticket data. """

    for id_ in tqdm(ids, desc="Load & store all tickets"):
        file_path = PATH / f"tickets{"_2000" if long_hist else ""}" / f"{id_}.parquet"
        result_path = PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq / f"{id_}_all_tickets.parquet"
        label_path = PATH / f"training_dataframes{"_2000" if long_hist else ""}" / freq / f"{id_}_labels.parquet"
        if file_path.exists(): #  and not result_path.exists():
            df = pd.read_parquet(file_path)
            df_labels = pd.read_parquet(label_path)
            start, end = df_labels.index[0], df_labels.index[-1]
            df = df[(df["start"] < end) & (df["end"]>=start)]
            if len(df):
                all_tickets = __create_binary_labels(df, freq=freq).reindex(df_labels.index , fill_value=0)
                all_tickets.to_parquet(result_path)
                print(f"{id_}: labels stored.")
            else:
                print(f"{id_}: empty DataFrame.")
        else:
            print(f"{id_} DataFrame not found.")


def preprocess_and_store_binary_labels(ids=ENERCON_IDS, freq="10min", long_hist = False):
    """Load and stores labels from given ticket data. """

    for id_ in tqdm(ids, desc="Load & store Labels"):
        file_path = PATH / f"tickets{"_2000" if long_hist else ""}" / f"{id_}_filtered.parquet"
        result_path = PATH / f"labels{"_2000" if long_hist else ""}" / f"{freq}_binary" / f"{id_}.parquet"
        if file_path.exists(): #  and not result_path.exists():
            df = pd.read_parquet(file_path)
            if len(df):
                labels = __create_binary_labels(df, freq=freq)
                labels.to_parquet(result_path)
                print(f"{id_}: labels stored.")
            else:
                print(f"{id_}: empty DataFrame.")
        else:
            print(f"{id_} DataFrame not found.")

# auxilliary
def load_all_unfiltered_tickets_from_ids(ids=ENERCON_IDS, long_hist=False):
    dfs = []
    for id_ in tqdm(ids, desc="Load all enercon tickets"):
        try:
            file_path = PATH/ f"tickets{"_2000" if long_hist else ""}" / f"{id_}.parquet"
            try:
                df = pd.read_parquet(file_path)
            except:
                if long_hist:
                    start = pd.Timestamp(2000, 1, 1, tzinfo=CET)
                else:
                    start= pd.Timestamp(2022, 10, 1, tzinfo=CET)
                end = pd.Timestamp(2024, 12, 31, tzinfo=CET)
                df = __get_ticket_data(start, end, id_)
                df.to_parquet(file_path)
            dfs.append(df)
        except:
            print("Coult not get data for ", id_)
    return pd.concat(dfs).rename(columns={"id":"ticket_id"})

