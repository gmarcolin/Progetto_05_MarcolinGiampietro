from datetime import datetime, timezone
import os
import time
import pandas as pd
import requests
import yaml
from src.data_logic import apply_drift, get_stream_batches, load_raw_partitions
from src.evaluate import evaluate_batch
from src.metrics_logger import log_batch_metrics, upload_retrain_snapshot

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def wait_for_retraining_completion(repo: str, token: str, timeout: int = 600, interval: int = 20):
    """
    Esegue polling verso GitHub Actions finché il workflow di retraining non si conclude.
    Restituisce (completed: bool, is_promoted: bool).
    """
    print(f"\n[Polling] In attesa del completamento del workflow di retraining su {repo}...")

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    runs_url = f"https://api.github.com/repos/{repo}/actions/runs"

    start_time = time.time()
    # Breve pausa iniziale per dare tempo a GitHub di avviare il job
    time.sleep(10)

    while time.time() - start_time < timeout:
        try:
            res = requests.get(runs_url, headers=headers, params={"per_page": 5})
            if res.status_code == 200:
                runs = res.json().get("workflow_runs", [])
                if runs:
                    latest_run = runs[0]
                    status = latest_run.get("status")
                    conclusion = latest_run.get("conclusion")
                    run_id_gh = latest_run.get("id")

                    print(f"[Polling] Workflow #{run_id_gh} -> Stato: {status} | Esito: {conclusion}")

                    if status == "completed":
                        # Interroghiamo i singoli job del workflow per vedere se deploy_champion è stato eseguito
                        jobs_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id_gh}/jobs"
                        jobs_res = requests.get(jobs_url, headers=headers)
                        is_promoted = False

                        if jobs_res.status_code == 200:
                            jobs = jobs_res.json().get("jobs", [])
                            for j in jobs:
                                # Se il job di deploy è terminato con successo, il modello è stato promosso
                                if "Deploy" in j.get("name", "") and j.get("conclusion") == "success":
                                    is_promoted = True
                                    break

                        print(f"[Polling] Retraining completato con esito: {conclusion}.") 
                        print(f"Modello Promosso a Champion: {is_promoted}")
                        # Attesa di cortesia per consentire a Hugging Face di propagare i nuovi pesi
                        time.sleep(10)
                        return True, is_promoted
            else:
                print(f"[Polling Warning] GitHub API status: {res.status_code}")
        except Exception as e:
            print(f"[Polling Exception] {e}")

        time.sleep(interval)

    print("[Polling Timeout] Tempo massimo raggiunto prima della conferma del retraining.")
    return False, False


def run_stream_monitoring():
    stream_df, _ = load_raw_partitions()
    batches = get_stream_batches(stream_df, n_chunks=config["data"]["backtest_chunks"])
    model_id = config["model"]["hub_id"]
    threshold_f1 = config["model"]["threshold_f1"]
    drift_schedule = config["monitoring"].get("drift_schedule", {})
    polling_cfg = config["monitoring"].get("polling", {})

    token_gh = os.getenv("GITHUB_TOKEN")
    repo_gh = os.getenv("GITHUB_REPOSITORY")

    buffer_list = []
    print(f"\n================ AVVIO STREAM CONTINUO ({len(batches)} BATCH) ================")

    for idx, batch in enumerate(batches):
        batch_num = idx + 1
        batch_id = f"batch_{batch_num:02d}"

        # 1. Applicazione del drift da dizionario (se configurato per questo batch)
        if batch_num in drift_schedule:
            drift_params = drift_schedule[batch_num]
            current_batch = apply_drift(
                batch,
                target_class=drift_params["target_class"],
                ratio=drift_params["imbalance_ratio"],
            )
            is_drifted = True
            event_type = f"drift_injected_r{drift_params['imbalance_ratio']}"
            print(f"\n⚡ [{batch_id}] APPLICAZIONE DRIFT: ratio={drift_params['imbalance_ratio']}")
        else:
            current_batch = batch
            is_drifted = False
            event_type = "production_stream"

        # 2. Accumulo del batch nella coda corrente
        buffer_list.append(current_batch)
        total_buffered_records = sum(len(b) for b in buffer_list)

        # 3. Valutazione batch
        f1, dist = evaluate_batch(model_id, current_batch)
        print(
            f"[{batch_id}] F1: {f1:.4f} | Buffer: {total_buffered_records} righe | "
            f"Pos: {dist['pos_pct']}% Neu: {dist['neu_pct']}% Neg: {dist['neg_pct']}%"
        )

        log_batch_metrics(
            batch_id=batch_id,
            model_id=model_id,
            f1_score=f1,
            distribution=dist,
            is_drifted=is_drifted,
            event_type=event_type,
        )

        # 4. Controllo del Model Decay
        if f1 < threshold_f1:
            run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            print(f"\n🚨 [ALERT] Performance sotto soglia su {batch_id} (F1: {f1:.4f} < {threshold_f1})")

            # A. Serializzazione e upload dello snapshot con tutti i batch accumulati finora
            accumulated_buffer = pd.concat(buffer_list, ignore_index=True)
            print(f"[MLOps] Snapshot {run_id} creato con {len(accumulated_buffer)} record.")
            upload_retrain_snapshot(accumulated_buffer, run_id=run_id)

            log_batch_metrics(
                batch_id=f"trigger_{batch_id}",
                model_id=model_id,
                f1_score=f1,
                distribution=dist,
                is_drifted=True,
                event_type="retrain_triggered",
                run_id=run_id,
            )

            # B. Invio del trigger di retraining a GitHub Actions
            if token_gh and repo_gh:
                url_dispatch = f"https://api.github.com/repos/{repo_gh}/dispatches"
                headers_dispatch = {
                    "Authorization": f"token {token_gh}",
                    "Accept": "application/vnd.github.v3+json",
                }
                payload = {"event_type": "model_decay", "client_payload": {"run_id": run_id}}
                requests.post(url_dispatch, json=payload, headers=headers_dispatch)
                print(f"[Dispatcher] Segnale inviato a GitHub Actions con run_id: {run_id}")

                # C. Polling con controllo dell'esito del Gatekeeper
                is_promoted = False
                _, is_promoted = wait_for_retraining_completion(
                    repo=repo_gh,
                    token=token_gh,
                    timeout=polling_cfg.get("timeout_seconds", 600),
                    interval=polling_cfg.get("interval_seconds", 20),
                )
            else:
                print("[Dispatcher Warning] GITHUB_TOKEN/GITHUB_REPOSITORY mancanti: salto il polling.")

            # D. GESTIONE BUFFER CONDIZIONALE (Solo su promozione!)
            if is_promoted:
                print(f"\n🎉 [MLOps] Nuovo Champion promosso in produzione!")
                print(f"[MLOps] Svuotamento buffer ({len(accumulated_buffer)} righe archiviate nello snapshot).")
                buffer_list = []  # <-- SI SVUOTA SOLO SE IL NUOVO MODELLO È STATO PROMOSSO
                print("[MLOps] Buffer azzerato. Ripresa del monitoraggio con il nuovo Champion...\n")
            else:
                print(f"\n⚠️ [MLOps] Challenger scartato dal Gatekeeper (non batte il Champion).")
                print(f"[MLOps] Il buffer NON viene svuotato ({len(buffer_list)} batch mantenuti per i prossimi cicli).")
                print("[MLOps] Il monitoraggio prosegue continuando ad accumulare dati...\n")

    print("\n================ SIMULAZIONE COMPLETATA SU TUTTI I BATCH ================")


if __name__ == "__main__":
    run_stream_monitoring()
