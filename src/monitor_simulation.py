from datetime import datetime
import os
import pandas as pd
import requests
import yaml
from src.data_logic import apply_drift, get_stream_batches, load_raw_partitions
from src.evaluate import evaluate_batch
from src.metrics_logger import log_batch_metrics, upload_retrain_snapshot

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def run_stream_monitoring():
    stream_df, _ = load_raw_partitions()
    batches = get_stream_batches(stream_df, n_chunks=config["data"]["backtest_chunks"])
    model_id = config["model"]["hub_id"]

    buffer_list = []
    print("\n--- AVVIO BACKTESTING STREAM PRODUZIONE (CHAMPION v0) ---")

    for idx, batch in enumerate(batches):
        batch_id = f"batch_{idx + 1}"
        is_last_batch = idx == len(batches) - 1

        # Per simulare il drift, corrompiamo l'ultimo batch
        if is_last_batch:
            current_batch = apply_drift(
                batch,
                target_class=config["monitoring"]["target_class_to_drop"],
                ratio=config["monitoring"]["imbalance_ratio"],
            )
            is_drifted = True
            event_type = "drift_injected"
        else:
            current_batch = batch
            is_drifted = False
            event_type = "production_stream"

        # Accumuliamo il batch nel buffer dal deploy
        buffer_list.append(current_batch)

        f1, dist = evaluate_batch(model_id, current_batch)
        print(f"[{batch_id}] F1: {f1:.4f} | Sentiment: Neg {dist['neg_pct']}% Neu {dist['neu_pct']}% Pos {dist['pos_pct']}%")

        log_batch_metrics(
            batch_id=batch_id,
            model_id=model_id,
            f1_score=f1,
            distribution=dist,
            is_drifted=is_drifted,
            event_type=event_type,
        )

        # Controllo soglia di performance
        if f1 < config["model"]["threshold_f1"]:
            run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n[ALERT] Model Decay rilevato su {batch_id} (F1: {f1:.4f} < {config['model']['threshold_f1']})")
            print(f"[MLOps] Generazione snapshot retrain con run_id: {run_id}...")

            accumulated_buffer = pd.concat(buffer_list, ignore_index=True)
            upload_retrain_snapshot(accumulated_buffer, run_id=run_id)

            # Log dell'evento di trigger
            log_batch_metrics(
                batch_id=f"trigger_{batch_id}",
                model_id=model_id,
                f1_score=f1,
                distribution=dist,
                is_drifted=True,
                event_type="retrain_triggered",
                run_id=run_id,
            )

            trigger_github_retrain(run_id)
            break


def trigger_github_retrain(run_id: str):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        print("[Dispatcher] GITHUB_TOKEN o GITHUB_REPOSITORY non impostati. Dispatch saltato.")
        return

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    payload = {"event_type": "model_decay", "client_payload": {"run_id": run_id}}
    res = requests.post(url, json=payload, headers=headers)
    if res.status_code == 204:
        print(f"[Dispatcher] Segnale 'model_decay' inviato a GitHub Actions con run_id: {run_id}")
    else:
        print(f"[Dispatcher Error] {res.status_code}: {res.text}")


if __name__ == "__main__":
    run_stream_monitoring()
