import pandas as pd
from datasets import load_dataset
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_validate_data(subset=None):
    dataset = load_dataset("mteb/tweet_sentiment_extraction")
    train_df = dataset['train'].to_pandas().dropna(subset=['text', 'label'])
    test_df = dataset['test'].to_pandas().dropna(subset=['text', 'label'])
    
    if subset:
        train_df = train_df.head(subset)
    
    valid_labels = [0, 1, 2]
    train_df = train_df[train_df['label'].isin(valid_labels)]
    return train_df, test_df

def get_backtest_chunks(df, n_chunks=5):
    return np.array_split(df, n_chunks)

def apply_imbalance(df, target_class=2, ratio=0.9):
    to_keep = df[df['label'] != target_class]
    to_corrupt = df[df['label'] == target_class].sample(frac=1-ratio)
    return pd.concat([to_keep, to_corrupt]).sample(frac=1)