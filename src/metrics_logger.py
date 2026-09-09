import os
from datetime import datetime
import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

METRICS_FILENAME = "metrics_history.csv"
REPO_ID = config["monitoring"]["metrics_dataset_id"]


def log_metric(batch_id: str, model_id: str, f1_score: float, is_drifted: bool, event_type: str):
    token = os.getenv("HF_TOKEN")
    api = HfApi()

    # 1. Recupera la cronologia esistente da Hugging Face (se presente)
    df_existing = pd.DataFrame()
    if token:
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=METRICS_FILENAME,
                repo_type="dataset",
                token=token,
            )
            df_existing = pd.read_csv(local_path)
        except Exception:
            # Primo avvio o dataset vuoto
            df_existing = pd.DataFrame()

    # 2. Crea la nuova riga
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_id": batch_id,
        "model_id": model_id,
        "f1_score": round(float(f1_score), 4),
        "is_drifted": bool(is_drifted),
        "event_type": event_type,
    }
    df_updated = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)

    # 3. Salva localmente come file temporaneo
    df_updated.to_csv(METRICS_FILENAME, index=False)
    print(f"[Logger] Registrata metrica: {row}")

    # 4. Push su Hugging Face Dataset Hub
    if token:
        try:
            api.upload_file(
                path_or_fileobj=METRICS_FILENAME,
                path_in_repo=METRICS_FILENAME,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=token,
                commit_message=f"log: metric update {batch_id} - F1: {row['f1_score']}",
            )
            print(f"[Logger] Metriche sincronizzate con HF Dataset: {REPO_ID}")
        except Exception as e:
            print(f"[Logger WARNING] Impossibile caricare su HF: {e}")
