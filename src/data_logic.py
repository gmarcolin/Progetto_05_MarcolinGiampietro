from datetime import datetime, timedelta
import logging
from datasets import load_dataset
import pandas as pd
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_partitions():
    """
    Carica il dataset, assegna a ciascun blocco temporale event_date
    e separa lo stream di produzione dal Golden Benchmark.
    """
    dataset = load_dataset(config["data"]["dataset_name"])
    valid_labels = [0, 1, 2]

    # 1. Train split: usato come flusso temporale di produzione e buffer di retraining
    stream_df = dataset["train"].to_pandas().dropna(subset=["text", "label"])
    stream_df = stream_df[stream_df["label"].isin(valid_labels)]
    stream_df = stream_df[stream_df["text"].str.len() > 2].reset_index(drop=True)

    # Assegnazione event_date simulata sui chunks
    n_chunks = config["data"]["backtest_chunks"]
    start_dt = datetime.strptime(config["data"].get("start_date", "2026-09-01"), "%Y-%m-%d")

    k, m = divmod(len(stream_df), n_chunks)
    chunk_dates = []
    for i in range(n_chunks):
        chunk_len = (i + 1) * k + min(i + 1, m) - (i * k + min(i, m))
        dt_str = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
        chunk_dates.extend([dt_str] * chunk_len)

    stream_df["event_date"] = chunk_dates

    # 2. Test split: congelato come Golden Benchmark Test Set
    golden_test_df = dataset["test"].to_pandas().dropna(subset=["text", "label"])
    golden_test_df = golden_test_df[golden_test_df["label"].isin(valid_labels)].reset_index(drop=True)

    print(f"Stream set caricato: {len(stream_df)} righe | Golden Test Set: {len(golden_test_df)} righe")

    return stream_df, golden_test_df


def get_batch_by_date(stream_df: pd.DataFrame, event_date: str) -> pd.DataFrame:
    """Restituisce il batch associato a una determinata event_date."""
    return stream_df[stream_df["event_date"] == event_date].copy().reset_index(drop=True)


def get_buffer_since_date(stream_df: pd.DataFrame, from_date: str, to_date: str) -> pd.DataFrame:
    """Restituisce i batch accumulati tra due date per il retraining."""
    mask = (stream_df["event_date"] >= from_date) & (stream_df["event_date"] <= to_date)
    return stream_df[mask].copy().reset_index(drop=True)


def apply_drift(batch_df: pd.DataFrame, target_class: int = 2, ratio: float = 0.92) -> pd.DataFrame:
    """Simula il drift sbilanciando fortemente la classe positiva."""
    to_keep = batch_df[batch_df["label"] != target_class]
    to_corrupt = batch_df[batch_df["label"] == target_class].sample(frac=1 - ratio, random_state=42)
    corrupted = pd.concat([to_keep, to_corrupt]).sample(frac=1, random_state=42).reset_index(drop=True)
    return corrupted


def get_smoke_test_data(golden_test_df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Estrae un piccolo campione dal Golden Test Set per la CI."""
    return golden_test_df.head(n).copy()
