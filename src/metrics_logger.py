from datetime import datetime, timezone
import os
from huggingface_hub import HfApi, hf_hub_download
import pandas as pd
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MONITORING_REPO = config["monitoring"]["monitoring_dataset_id"]
METRICS_PATH_REMOTE = "metrics/metrics_history.csv"
METRICS_LOCAL = "metrics_history.csv"


def log_batch_metrics(
    batch_id: str,
    model_id: str,
    f1_score: float,
    distribution: dict,
    is_drifted: bool,
    event_type: str,
    run_id: str = "none",
):
    """Salva la metrica nella configurazione 'metrics/' di sentiment-analysis-monitoring."""
    token = os.getenv("HF_TOKEN")
    api = HfApi()

    # 1. Recupera la cronologia esistente da Hugging Face (se presente)
    df_existing = pd.DataFrame()
    if token:
        try:
            local_path = hf_hub_download(
                repo_id=MONITORING_REPO,
                filename=METRICS_PATH_REMOTE,
                repo_type="dataset",
                token=token,
            )
            df_existing = pd.read_csv(local_path)
        except Exception:
            # Primo avvio o dataset vuoto
            df_existing = pd.DataFrame()

    # 2. Crea la nuova riga
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "batch_id": batch_id,
        "run_id": run_id,
        "model_id": model_id,
        "f1_score": round(float(f1_score), 4),
        "neg_pct": distribution.get("neg_pct", 0.0),
        "neu_pct": distribution.get("neu_pct", 0.0),
        "pos_pct": distribution.get("pos_pct", 0.0),
        "is_drifted": bool(is_drifted),
        "event_type": event_type,
    }

    df_updated = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)

    # 3. Salva localmente come file temporaneo
    df_updated.to_csv(METRICS_LOCAL, index=False)
    print(f"[Logger] Registrata metrica: {row}")

    # 4. Push su Hugging Face Dataset Hub
    if token:
        try:
            api.upload_file(
                path_or_fileobj=METRICS_LOCAL,
                path_in_repo=METRICS_PATH_REMOTE,
                repo_id=MONITORING_REPO,
                repo_type="dataset",
                token=token,
                commit_message=f"telemetry: log batch {batch_id} (F1: {row['f1_score']})",
            )
            print(f"[Logger] Metriche salvate in {MONITORING_REPO}/{METRICS_PATH_REMOTE}")
        except Exception as e:
            print(f"[Logger WARNING] Upload metriche su HF fallito: {e}")


def upload_retrain_snapshot(buffer_df: pd.DataFrame, run_id: str):
    """Salva i dati nella configurazione 'data/' di sentiment-analysis-monitoring."""
    token = os.getenv("HF_TOKEN")
    api = HfApi()

    local_snapshot = f"retrain_{run_id}.parquet"
    buffer_df.to_parquet(local_snapshot, index=False)

    if token:
        try:
            # 1. Snapshot immutabile archiviato con run_id
            api.upload_file(
                path_or_fileobj=local_snapshot,
                path_in_repo=f"data/snapshots/retrain_{run_id}.parquet",
                repo_id=MONITORING_REPO,
                repo_type="dataset",
                token=token,
                commit_message=f"lineage: snapshot {run_id} ({len(buffer_df)} record)",
            )
            # 2. Puntatore aggiornato per il job di retraining
            api.upload_file(
                path_or_fileobj=local_snapshot,
                path_in_repo="data/latest_retrain_buffer.parquet",
                repo_id=MONITORING_REPO,
                repo_type="dataset",
                token=token,
                commit_message=f"lineage: update data/latest_retrain_buffer.parquet ({run_id})",
            )
            print(f"[Lineage] Dati di retrain salvati in {MONITORING_REPO}/data/")
        except Exception as e:
            print(f"[Lineage WARNING] Upload snapshot fallito: {e}")

    return local_snapshot
