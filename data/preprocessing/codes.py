# %%
from pathlib import Path
import os
os.chdir("C:/Users/Paul.Wecker/dev/Studies/predictive_maintenance/data/preprocessing/")
import re

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from powerdata.data.model import create_session, EnergySystem, ManufacturerStatus, Select
from data_registry import ENERCON_IDS, PATH

def create_cluster_wordclouds(df, text_column='info', cluster_column='cluster', 
                             figsize=(15, 10), max_words=100, method='tfidf'):

    def preprocess_text(text):
        """Clean and preprocess text for word cloud generation."""
        if pd.isna(text):
            return ""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    # Get unique clusters
    clusters = sorted(df[cluster_column].unique())
    n_clusters = len(clusters)
    
    # Calculate subplot dimensions
    cols = min(3, n_clusters)  # Max 3 columns
    rows = (n_clusters + cols - 1) // cols
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_clusters == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    wordcloud_objects = {}
    
    for i, cluster_id in enumerate(clusters):
        # Get text for this cluster
        cluster_texts = df[df[cluster_column] == cluster_id][text_column].tolist()
        
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in cluster_texts]
        combined_text = ' '.join(processed_texts)
        
        if not combined_text.strip():
            # Handle empty cluster
            axes[i].text(0.5, 0.5, f'Cluster {cluster_id}\n(No valid text)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Cluster {cluster_id} (Empty)')
            axes[i].axis('off')
            continue
        
        # Generate word frequencies based on method
        if method == 'tfidf':
            # Use TF-IDF to find important words across clusters
            all_cluster_texts = []
            for cid in clusters:
                cluster_docs = df[df[cluster_column] == cid][text_column].tolist()
                cluster_doc = ' '.join([preprocess_text(text) for text in cluster_docs])
                all_cluster_texts.append(cluster_doc)
            
            # Fit TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=max_words, stop_words='english', 
                                       ngram_range=(1, 2), min_df=1)
            tfidf_matrix = vectorizer.fit_transform(all_cluster_texts)
            
            # Get TF-IDF scores for current cluster
            cluster_idx = list(clusters).index(cluster_id)
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix[cluster_idx].toarray()[0]
            
            # Create word frequency dictionary
            word_freq = dict(zip(feature_names, tfidf_scores))
            # Filter out zero scores
            word_freq = {word: score for word, score in word_freq.items() if score > 0}
        
        else:  # frequency method
            # Simple word frequency counting
            words = combined_text.split()
            # Remove common stop words manually (basic set)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                         'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                         'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            words = [word for word in words if word not in stop_words and len(word) > 2]
            word_freq = dict(Counter(words).most_common(max_words))
        
        if not word_freq:
            axes[i].text(0.5, 0.5, f'Cluster {cluster_id}\n(No significant words)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Cluster {cluster_id} (No Words)')
            axes[i].axis('off')
            continue
        
        # Create word cloud
        wordcloud = WordCloud(width=400, height=300, 
                             background_color='white',
                             max_words=max_words,
                             colormap='viridis',
                             relative_scaling=0.5,
                             random_state=42).generate_from_frequencies(word_freq)
        
        # Plot word cloud
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'Cluster {cluster_id} ({len(cluster_texts)} items)', 
                         fontsize=12, fontweight='bold')
        axes[i].axis('off')
        
        # Store wordcloud object
        wordcloud_objects[cluster_id] = wordcloud
    
    # Hide empty subplots
    for i in range(n_clusters, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return wordcloud_objects


def save_cluster_wordclouds(wordcloud_objects, output_dir='wordclouds'):
    """
    Save individual word clouds to files.
    
    Parameters:
    -----------
    wordcloud_objects : dict
        Dictionary of WordCloud objects from create_cluster_wordclouds
    output_dir : str
        Directory to save the word cloud images
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for cluster_id, wordcloud in wordcloud_objects.items():
        filename = f"cluster_{cluster_id}_wordcloud.png"
        filepath = os.path.join(output_dir, filename)
        wordcloud.to_file(filepath)
        print(f"Saved word cloud for cluster {cluster_id} to {filepath}")


def cluster_embeddings(embeddings, infos, codes, n_clusters=5):
    normalized_embeddings = normalize(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_embeddings)
    
    df = pd.DataFrame({"cluster": cluster_labels, "info":infos, "codes": codes}).sort_values(by="cluster").reset_index()
    return df, kmeans


def create_elbow_plot(embeddings, max_clusters=10):
    # Normalize the embeddings to unit length
    normalized_embeddings = normalize(embeddings)
    
    # Calculate inertia (within-cluster sum of squares) for different numbers of clusters
    inertia = []
    K_range = range(1, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(normalized_embeddings)
        inertia.append(kmeans.inertia_)
    
    # Create the elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Plot for K-means Clustering')
    plt.xticks(K_range)
    plt.grid(True)
    plt.show()


def plot_silhoette():
    df = pd.read_csv(PATH / f"training_dataframes_2000" / "code_counts_infos.csv")

    infos = df["info"].to_list()
    embeddings = create_bert_embeddings(infos)
    embeddings = normalize(embeddings)
    model = KMeans(42, random_state=42)
    X = np.array(embeddings, dtype=np.float32)
    X = np.ascontiguousarray(X, dtype=np.float32)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

    visualizer.fit(X)
    visualizer.show()


def add_cluster_assignments_to_stats(long_hist, Ns):
    df = pd.read_csv(PATH / f"training_dataframes_2000" / "code_counts_infos.csv")
    codes = df["error_code"].astype(str).to_list()
    infos = df["info"].to_list()
    embeddings = create_bert_embeddings(infos)
    df = pd.read_csv(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / "code_counts_infos.csv")
    for n in Ns:
        df_cluster, _ = cluster_embeddings(embeddings, infos, codes, n_clusters=n)
        df_cluster = df_cluster[["cluster", "codes"]]
        df_cluster = df_cluster.astype(int)
        df_cluster = df_cluster.rename(columns={"cluster":f"cluster_N{n}", "codes":"error_code"})
        df = pd.merge(df, df_cluster, left_on="error_code", right_on="error_code")
    return df


def create_kmeans_assignments(n, long_hist):
    code_counts = pd.read_csv(PATH / f"training_dataframes{"_2000" if long_hist else ""}" / "code_counts_infos.csv")
    codes = code_counts["error_code"].astype(str).to_list()
    infos = code_counts["info"].to_list()
    embeddings = create_bert_embeddings(infos)
    return cluster_embeddings(embeddings, infos, codes, n_clusters=n)


def merge_code_columns_by_cluster(df_codes, cluster_assignments, n):
    cluster_assignments["error_code"] = cluster_assignments["error_code"].astype(float).astype(str)
    cluster_assignments = cluster_assignments[cluster_assignments["error_code"].isin(df_codes.columns)]
    for c in range(0, n):
        code_cols = cluster_assignments[cluster_assignments[f"cluster_N{n}"]==c]["error_code"].unique()
        print(len(code_cols))        
        if len(code_cols) > 0:
            # Existing sum column
            df_codes[f"code_cluster_{c}"] = df_codes[code_cols].sum(axis=1)
            
            # State changes column
            code_diffs = df_codes[code_cols].diff()  # diff for each code
            code_diffs_abs = code_diffs.abs()        # convert -1 to +1
            df_codes[f"code_cluster_{c}_changes"] = code_diffs_abs.sum(axis=1)
        else:
            # Handle empty clusters
            df_codes[f"code_cluster_{c}"] = 0
            df_codes[f"code_cluster_{c}_changes"] = 0
    cluster_cols = []
    for c in range(0, n):
        cluster_cols.extend([f"code_cluster_{c}", f"code_cluster_{c}_changes"])
    
    df_codes = df_codes[cluster_cols]
    df_codes = df_codes.astype(int)
    return df_codes


def create_bert_embeddings(infos):
    embeddings = get_bert_embeddings(infos)
    return embeddings


def get_bert_embeddings(texts):
    local_model_path = "C:/Users/Paul.Wecker/dev/Studies/predictive_maintenance/data/preprocessing/BERT"
    
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    model = BertModel.from_pretrained(local_model_path)
    with torch.no_grad():
        embeddings = []
        for text in tqdm(texts, desc="compute embedding"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embedding[0])
        return embeddings


def __get_codes_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    es_id: int
) -> pd.DataFrame:
    """Create DataFrame with events data along software_id.

    Arguments:
        start: start of period of interest
        end: end of period of interest
        es_id: energy system ID

    Returns:
        DataFrame with cols: start, end, error_code, es_id and software_id
    """
    with create_session() as s:
        es = EnergySystem.by_id(s, es_id)

        scadas = es.notifications
        software_id = es.software_id
        df = pd.DataFrame([
            {
                "start": s.start,
                "end": s.end,
                "error_code": s.error_code,
                "es_id": es_id,
                "software_id": software_id
            } for s in scadas if ((s.start < end) and (s.start >= start))
        ])

    return df


def load_all_codes_with_info():
    """Loads DF with columns:
    - es_id: identifies system
    - start, end
    - duration
    - error_code: as number
    - info: text explaining the code"""

    df = pd.read_parquet(PATH / "EnerconOpcXmlDaCs82a" / "complete_codes_with_info.parquet")
    return df


def encode_codes_numpy(df: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """Create DataFrame with binary encodings for a given frequency from DataFrame.

    Arguments:
        - df: DataFrame with columns `start`, `end`, `error_code` and `es_id`
        - es_id identifying energy system
        - freq: string from which pandas.tseries.frequencies.Offset is created
    Returns:
        - DataFrame with pd.TimeIndex with given frequency, one column for each error code in the passed DataFrame.
    """

    freq_offset = pd.tseries.frequencies.to_offset(freq)
    freq_seconds = pd.Timedelta(freq_offset).total_seconds()
    df = df[~pd.isna(df["error_code"])].copy()

    error_codes = df["error_code"].unique()
    filtered_df = df[["start", "end", "error_code"]].copy()

    start_time = filtered_df["start"].min().floor("h")

    if not pd.isna(filtered_df["end"].max()):
        end_time = filtered_df["end"].max().ceil("h")
    else:
        end_time = filtered_df["start"].max().ceil("h")

    time_index = pd.date_range(start_time, end_time, freq=freq_offset)
    time_stamps_unix = np.array([ts.timestamp() for ts in time_index])
    result_array = np.zeros((len(time_index), len(error_codes)), dtype=np.int8)
    error_code_to_idx = {code: i for i, code in enumerate(error_codes)}
    starts_unix = np.array([ts.timestamp() for ts in filtered_df["start"]])
    ends = filtered_df["end"].copy()

    for i, end in enumerate(ends):
        if pd.isna(end):
            ends.iloc[i] = filtered_df["start"].iloc[i] + freq_offset

    ends_unix = np.array([ts.timestamp() for ts in ends])
    error_codes_array = np.array([error_code_to_idx[code] for code in filtered_df["error_code"]])

    for i, timestamp in enumerate(time_stamps_unix):
        prev_timestamp = timestamp - freq_seconds
        active_mask = (starts_unix <= timestamp) & (ends_unix > prev_timestamp)

        for j in np.where(active_mask)[0]:
            result_array[i, error_codes_array[j]] = 1

    result_df = pd.DataFrame(
        result_array,
        index=time_index,
        columns=error_codes,
        dtype=int
    )

    return result_df


def get_status(software_id: int=75) -> pd.DataFrame:
    """Get metadata on error codes for a software.
    Metadata consists of a boolean `register` which indicates whether this error code triggers an automatic message;
    `name` describing the error code, along with the software_id, the code itself (error_code) and its id.

    Arguments:
        software_id: identifies software

    Returns:
        DataFrame with columns `software_id`, `status` (error code), `id`, `name`, `register`.
    """
    with create_session() as s:
        s = create_session()

        query = (
            Select(ManufacturerStatus)
            .where(ManufacturerStatus.software_id == software_id)
        )
        statuses = s.scalars(query).all()
    df = pd.DataFrame.from_records([
        {
            "software_id": s.software_id,
            "status": s.status,
            "name": s.name,
            "register": s.register
        } for s in statuses
    ])
    df = df.rename(columns={"status": "error_code", "name": "info"})
    return df


def load_and_store_codes(start, end, ids=ENERCON_IDS) -> None:
    path = PATH / "codes_2000"
    for id_ in tqdm(ids, desc="codes:"):
        file_path = Path(path / f"{id_}.parquet")
        if not file_path.exists():
            print(f"Loading Codes for {id_}.")
            df = __get_codes_data(start, end, id_)

            if len(df):
                df.to_parquet(file_path)
                print(f"Added Code-Data for {id_}.")
            else:
                print(f"No Code-Data for {id_}.")
        else:
            print(f"Codes for {id_} found.")


def preprocess_and_store_one_hot_codes(ids = ENERCON_IDS, freq="10min", long_hist=True, overwrite=False):
    path = Path(PATH / f"codes{"_2000" if long_hist else ""}")
    for id_ in tqdm(ids, desc="Create OneHot Codes"):
        file_path = Path(path / f"{id_}.parquet")
        result_file_path = Path(path / f"{id_}_{freq}_onehot.parquet")
        if (not result_file_path.exists()) or overwrite:
            try:
                df = pd.read_parquet(file_path)
            except FileNotFoundError:
                print(f"Could not find data for id: {id_}")
            df_filtered = encode_codes_numpy(df, freq=freq)
            df_filtered.to_parquet(result_file_path)
            print(f"Stored Encoded Codes for id: {id_}")
        else:
            print(f"Found Encoded Codes for id: {id_}")


def load_enercon_codes_raw(ids=ENERCON_IDS) -> pd.DataFrame:
    dfs = []
    for id_ in tqdm(ids, desc="Loading Codes for Enercon Systems"):
        df_codes = pd.read_parquet(PATH  / "codes" / f"{id_}.parquet")
        dfs.append(df_codes)
    return pd.concat(dfs)


def plot_codes_per_subset_size(df: pd.DataFrame):
    """Create Histogram for maximum number of error codes shared across x systems.
    Arguments:
        - DataFrame with columns `es_id` and `error_code`
    Returns
        Figure with plot.
    """

    df = (df[["es_id", "error_code"]]
          .drop_duplicates()
          .groupby(by="error_code").count()
          .sort_values(by="es_id"))
    df["count"] = 1
    df = (
        df
        .groupby(by="es_id").count()
        .sort_index(ascending=False)
    )

    df["count"] = df["count"].cumsum()
    max_x = max(df.index.to_list())

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(df.index, df["count"], alpha=0.7, color="blue", label="Error Codes")
    ax.set_xlabel("Size of Energy System Subset")
    ax.set_ylabel("Number of Different Error Codes")
    ax.set_title("Number of Codes for Subset-Size of Systems")
    ax.set_xticks(np.arange(1, max_x + 1, step=max(1, max_x // 10)))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()

    return fig


def code_counts(ids=ENERCON_IDS):
    combined_value_counts = pd.Series()

    for id_ in tqdm(ids, desc="Getting Code Counts"):
        df = pd.read_parquet(PATH / "codes_2000" / f"{id_}.parquet")
        counts = df["info"].value_counts()
        combined_value_counts = combined_value_counts.add(counts, fill_value=0)
    return combined_value_counts.sort_values()
