from datetime import datetime
import json
import os
from airflow.models import Variable
import evaluate
import numpy as np
import pandas as pd
import requests
import yaml
from src.data_logic import apply_drift, get_batch_by_date, get_buffer_since_date, load_raw_partitions
from src.metrics_logger import log_batch_metrics, upload_retrain_snapshot

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

STAGING_DIR = "/tmp/airflow_staging"
os.makedirs(STAGING_DIR, exist_ok=True)


def ingest_batch_task(**context):
    """Task 1: Preleva il batch per la data corrente, inietta il drift se previsto e salva lo staging."""
    ds = context["ds"]
    stream_df, _ = load_raw_partitions()
    batch_df = get_batch_by_date(stream_df, ds)

    if batch_df.empty:
        raise ValueError(f"Nessun dato disponibile per event_date: {ds}")

    # Indice numerico del giorno per il drift schedule
    chunk_idx = int(ds.split("-")[-1])
    drift_cfg = config["monitoring"].get("drift_schedule", {}).get(chunk_idx)
    is_drifted = False

    if drift_cfg:
        print(f"[Ingest] Iniezione drift sul batch {chunk_idx}: ratio={drift_cfg['imbalance_ratio']}")
        batch_df = apply_drift(batch_df, target_class=drift_cfg["target_class"], ratio=drift_cfg["imbalance_ratio"])
        is_drifted = True

    staging_path = os.path.join(STAGING_DIR, f"batch_{ds}.parquet")
    batch_df.to_parquet(staging_path, index=False)

    # Salviamo metadati leggeri via XCom
    context["ti"].xcom_push(key="is_drifted", value=is_drifted)
    context["ti"].xcom_push(key="chunk_idx", value=chunk_idx)
    context["ti"].xcom_push(key="batch_size", value=len(batch_df))
    print(f"[Ingest] Batch per {ds} salvato in {staging_path} ({len(batch_df)} righe)")


def run_inference_task(**context):
    """Task 2: Invia i testi all'API FastAPI /predict e memorizza le risposte."""
    ds = context["ds"]
    api_url = os.environ.get("SENTIMENT_API_URL", config["monitoring"].get("api_url", "http://sentiment-api:8000"))

    staging_path = os.path.join(STAGING_DIR, f"batch_{ds}.parquet")
    batch_df = pd.read_parquet(staging_path)

    predict_url = f"{api_url}/predict"
    payload = {"texts": batch_df["text"].tolist()}

    print(f"[Inference] Inoltro {len(payload['texts'])} testi verso {predict_url}...")
    res = requests.post(predict_url, json=payload, timeout=120)

    if res.status_code != 200:
        raise RuntimeError(f"Chiamata a {predict_url} fallita ({res.status_code}): {res.text}")

    preds_data = res.json()
    preds_path = os.path.join(STAGING_DIR, f"preds_{ds}.json")
    with open(preds_path, "w") as f:
        json.dump(preds_data, f)

    print(f"[Inference] Ricevute {len(preds_data)} predizioni e salvate in {preds_path}")


def evaluate_and_log_task(**context):
    """Task 3: Calcola l'F1 pesato confrontando con le label reali e registra su HF."""
    ds = context["ds"]
    ti = context["ti"]

    batch_df = pd.read_parquet(os.path.join(STAGING_DIR, f"batch_{ds}.parquet"))
    with open(os.path.join(STAGING_DIR, f"preds_{ds}.json"), "r") as f:
        response_data = json.load(f)

    preds_data = response_data.get("predictions", [])
    model_id = response_data.get("model_id", "unknown_model")

    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    y_pred = [mapping.get(p["label"], 1) for p in preds_data]
    y_true = batch_df["label"].tolist()

    f1_metric = evaluate.load("f1", trust_remote_code=True)
    f1_score = f1_metric.compute(predictions=y_pred, references=y_true, average="weighted")["f1"]

    total = len(y_pred)
    dist = {
        "neg_pct": round(float(np.sum(np.array(y_pred) == 0) / total * 100), 2),
        "neu_pct": round(float(np.sum(np.array(y_pred) == 1) / total * 100), 2),
        "pos_pct": round(float(np.sum(np.array(y_pred) == 2) / total * 100), 2),
    }

    is_drifted = ti.xcom_pull(task_ids="ingest_batch", key="is_drifted")
    chunk_idx = ti.xcom_pull(task_ids="ingest_batch", key="chunk_idx")
    event_type = f"drift_batch_{chunk_idx}" if is_drifted else "stream_batch"

    log_batch_metrics(
        batch_id=f"batch_{ds}",
        model_id=model_id,
        f1_score=f1_score,
        distribution=dist,
        is_drifted=is_drifted,
        event_type=event_type,
    )

    print(f"[Evaluation] F1 = {f1_score:.4f} | Distribuzione: {dist}")
    # Esporta l'F1-score verso XCom per il branch operator
    ti.xcom_push(key="f1_score", value=f1_score)


def model_decay_branch_task(**context):
    """Task 4 (Branch): Valuta la soglia F1 e sceglie quale task eseguire dopo."""
    ti = context["ti"]
    f1_score = ti.xcom_pull(task_ids="evaluate_metrics", key="f1_score")
    threshold_f1 = config["model"]["threshold_f1"]

    print(f"[Branch Check] F1 rilevato: {f1_score:.4f} vs Soglia: {threshold_f1}")

    if f1_score < threshold_f1:
        print("[Branch Check] -> Bivio verso: trigger_retraining")
        return "trigger_retraining"
    else:
        print("[Branch Check] -> Bivio verso: decay_not_detected")
        return "decay_not_detected"


def trigger_retraining_task(**context):
    """Task 5a: Genera lo snapshot dei dati accumulati dal deploy e invia il dispatch a GitHub."""
    ds = context["ds"]
    last_promoted = Variable.get("last_promoted_event_date", default_var="2026-09-01")
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    print(f"[Retrain Trigger] Accumulo dati da {last_promoted} a {ds}...")
    stream_df, _ = load_raw_partitions()
    buffer_df = get_buffer_since_date(stream_df, from_date=last_promoted, to_date=ds)

    upload_retrain_snapshot(buffer_df, run_id=run_id)
    print(f"[Retrain Trigger] Snapshot {run_id} caricato su HF ({len(buffer_df)} record).")

    token_gh = os.getenv("GITHUB_TOKEN")
    repo_gh = os.getenv("GITHUB_REPOSITORY")

    if token_gh and repo_gh:
        dispatch_url = f"https://api.github.com/repos/{repo_gh}/dispatches"
        headers = {
            "Authorization": f"token {token_gh}",
            "Accept": "application/vnd.github.v3+json",
        }
        payload = {
            "event_type": "model_decay",
            "client_payload": {"run_id": run_id, "trigger_date": ds},
        }
        res = requests.post(dispatch_url, json=payload, headers=headers)
        if res.status_code == 204:
            print(f"[Retrain Trigger] Dispatch inviato a GitHub Actions per {run_id}.")
        else:
            raise RuntimeError(f"Errore dispatch ({res.status_code}): {res.text}")
    else:
        print("[Retrain Trigger Warning] GITHUB_TOKEN o GITHUB_REPOSITORY non impostati.")
