from datetime import datetime, timezone
import io
import json
import os
from zipfile import ZipFile
from airflow.exceptions import AirflowSkipException
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


def parse_label_to_id(val) -> int:
    if isinstance(val, int):
        return val
    val_str = str(val).lower().strip()
    mapping = {
        "negative": 0, "0": 0, "label_0": 0,
        "neutral": 1, "1": 1, "label_1": 1,
        "positive": 2, "2": 2, "label_2": 2,
    }
    return mapping.get(val_str, 1)


def ingest_batch_task(**context):
    """Task 1: Preleva il batch per la data corrente, inietta il drift se previsto e salva lo staging."""
    ds = context["ds"]
    stream_df, _ = load_raw_partitions()
    batch_df = get_batch_by_date(stream_df, ds)

    if batch_df.empty:
        raise ValueError(f"Nessun dato trovato per event_date: {ds}")

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

    y_pred = [parse_label_to_id(p["label"]) for p in preds_data]
    y_true = [parse_label_to_id(lbl) for lbl in batch_df["label"]]

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
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    print(f"[Retrain Trigger] Accumulo dati da {last_promoted} a {ds}...")
    stream_df, _ = load_raw_partitions()
    buffer_df = get_buffer_since_date(stream_df, from_date=last_promoted, to_date=ds)

    upload_retrain_snapshot(buffer_df, run_id=run_id)
    print(f"[Retrain Trigger] Snapshot {run_id} caricato su HF ({len(buffer_df)} record).")

    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")

    if token and repo:
        url = f"https://api.github.com/repos/{repo}/dispatches"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {
            "event_type": "model_decay",
            "client_payload": {"run_id": run_id}
        }
        res = requests.post(url, json=payload, headers=headers)
        if res.status_code == 204:
            print(f"[Airflow Task] Retrain trigger inviato a GitHub Actions per {run_id}.")
        else:
            raise RuntimeError(f"Errore dispatch ({res.status_code}): {res.text}")
    else:
        print("[Retrain Trigger Warning] GITHUB_TOKEN o GITHUB_REPOSITORY non impostati.")

    context["ti"].xcom_push(key="run_id", value=run_id)


def gatekeep_and_decide_task(**context):
    """
    Task Gatekeeper (puro, senza polling):
    Recupera l'artifact del retraining, valuta le metriche e decide se avviare il deploy.
    """
    ti = context["ti"]
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    retrain_run_id = ti.xcom_pull(task_ids="wait_retrain_sensor", key="retrain_run_id")
    run_id = ti.xcom_pull(task_ids="trigger_retraining", key="run_id")

    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    # Download eval_results.json
    artifacts_url = f"https://api.github.com/repos/{repo}/actions/runs/{retrain_run_id}/artifacts"
    art_res = requests.get(artifacts_url, headers=headers).json()
    artifact_id = next(a["id"] for a in art_res.get("artifacts", []) if a["name"] == "eval-results")

    zip_res = requests.get(
        f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip", headers=headers
    )
    with ZipFile(io.BytesIO(zip_res.content)) as z:
        eval_data = json.loads(z.read("eval_results.json"))

    cand_f1 = eval_data["candidate_f1"]
    champ_f1 = eval_data["champion_f1"]
    threshold = config["model"]["threshold_f1"]

    print(f"[Gatekeeper Policy] Challenger F1: {cand_f1} vs Champion F1: {champ_f1} (Soglia: {threshold})")

    if cand_f1 >= champ_f1 and cand_f1 >= threshold:
        print("[Gatekeeper Policy] APPROVATO: Innesco workflow Deploy...")
        requests.post(
            f"https://api.github.com/repos/{repo}/dispatches",
            headers=headers,
            json={
                "event_type": "model_approved",
                "client_payload": {"run_id": run_id, "retrain_run_id": retrain_run_id},
            },
        )
        ti.xcom_push(key="deploy_approved", value=True)
    else:
        print("[Gatekeeper Policy] RESPINTO: Challenger non conforme.")
        ti.xcom_push(key="deploy_approved", value=False)
        raise AirflowSkipException("Challenger respinto dal Gatekeeper. Deploy saltato.")


def hot_reload_and_verify_task(**context):
    """
    Task Hot-Reload e Rollback (puro, senza polling):
    Scarica il commit_sha approvato, aggiorna l'API ed esegue rollback se il controllo /health fallisce.
    """
    ti = context["ti"]
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    deploy_run_id = ti.xcom_pull(task_ids="wait_deploy_sensor", key="deploy_run_id")
    api_url = os.environ.get("SENTIMENT_API_URL", config["monitoring"].get("api_url", "http://sentiment-api:8000"))

    # Recupero vecchio SHA prima dell'aggiornamento
    initial_health = requests.get(f"{api_url}/health").json()
    old_revision = initial_health.get("model_revision", "base-fallback")

    # Download deploy_result.json
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    artifacts_url = f"https://api.github.com/repos/{repo}/actions/runs/{deploy_run_id}/artifacts"
    art_res = requests.get(artifacts_url, headers=headers).json()
    artifact_id = next(a["id"] for a in art_res.get("artifacts", []) if a["name"] == "deploy-result")

    zip_res = requests.get(
        f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip", headers=headers
    )
    with ZipFile(io.BytesIO(zip_res.content)) as z:
        deploy_data = json.loads(z.read("deploy_result.json"))

    new_revision = deploy_data["commit_sha"]

    print(f"[Hot-Reload] Richiesta POST /reload?revision={new_revision}...")
    requests.post(f"{api_url}/reload", params={"revision": new_revision})

    # Verifica dello stato
    health = requests.get(f"{api_url}/health").json()
    if health.get("status") == "healthy" and health.get("model_revision") == new_revision:
        print(f"🎉 [Deploy Success] Hot-reload verificato: servizio su revisione {new_revision}")
        Variable.set("last_promoted_event_date", context["ds"])
    else:
        print(f"⚠️ [Deploy Warning] Nuovo modello instabile. Innesco Rollback a {old_revision}...")
        requests.post(f"{api_url}/reload", params={"revision": old_revision})
        health_rb = requests.get(f"{api_url}/health").json()
        if health_rb.get("model_revision") == old_revision:
            raise RuntimeError(f"ALERT (Warning): Rilascio fallito ma Rollback riuscito a {old_revision}.")
        else:
            raise SystemError("CRITICAL ALERT (P1): Servizio di produzione compromesso. Fallito anche il rollback!")
