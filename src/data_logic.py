import logging
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_partitions():
    """Carica il dataset MTEB e separa lo stream di produzione dal Golden Benchmark."""
    dataset = load_dataset("mteb/tweet_sentiment_extraction")

    # Train split: usato come flusso temporale di produzione e buffer di retraining
    stream_df = dataset["train"].to_pandas().dropna(subset=["text", "label"])
    valid_labels = [0, 1, 2]
    stream_df = stream_df[stream_df["label"].isin(valid_labels)]
    stream_df = stream_df[stream_df["text"].str.len() > 2].reset_index(drop=True)

    # Test split: congelato come Golden Benchmark Test Set
    golden_test_df = dataset["test"].to_pandas().dropna(subset=["text", "label"])
    golden_test_df = golden_test_df[golden_test_df["label"].isin(valid_labels)].reset_index(
        drop=True
    )

    logger.info(f"Stream set caricato: {len(stream_df)} righe | Golden Test Set: {len(golden_test_df)} righe")
    return stream_df, golden_test_df


def get_stream_batches(df, n_chunks=5):
    """Divide il flusso in batch sequenziali ordinati per il backtesting temporale."""
    k, m = divmod(len(df), n_chunks)
    return [
        df.iloc[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)].copy().reset_index(drop=True)
        for i in range(n_chunks)
    ]


def apply_drift(batch_df, target_class=2, ratio=0.92):
    """Simula il drift sbilanciando fortemente la classe positiva."""
    to_keep = batch_df[batch_df["label"] != target_class]
    to_corrupt = batch_df[batch_df["label"] == target_class].sample(frac=1 - ratio, random_state=42)
    corrupted = pd.concat([to_keep, to_corrupt]).sample(frac=1, random_state=42).reset_index(drop=True)
    return corrupted


def get_smoke_test_data(golden_test_df, n=25):
    """Estrae un piccolo campione dal Golden Test Set per la CI."""
    return golden_test_df.head(n).copy()
