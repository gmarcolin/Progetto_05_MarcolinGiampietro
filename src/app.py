import os
import gradio as gr
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from src.metrics_logger import METRICS_FILENAME

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

REPO_ID = config["monitoring"]["metrics_dataset_id"]


def load_metrics_data():
    token = os.getenv("HF_TOKEN")
    try:
        # Scarica sempre l'ultima versione dal Dataset Hub
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=METRICS_FILENAME,
            repo_type="dataset",
            token=token,
        )
        df = pd.read_csv(path)
        latest_f1 = df["f1_score"].iloc[-1]
        status = f"Ultima F1: {latest_f1:.4f} | Eventi totali: {len(df)} (Sorgente: HF Dataset)"
        return df, status
    except Exception as e:
        # Fallback se il dataset è ancora vuoto o non configurato
        empty_df = pd.DataFrame(
            columns=["timestamp", "batch_id", "model_id", "f1_score", "is_drifted", "event_type"]
        )
        return empty_df, f"Nessun dato remoto disponibile ({e})"
