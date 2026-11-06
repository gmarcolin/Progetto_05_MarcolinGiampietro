import json
import os
from huggingface_hub import hf_hub_download
import pandas as pd
import yaml
from src.data_logic import load_raw_partitions
from src.evaluate import champion_challenger_check
from src.train import train_model

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def run_retrain_job(run_id: str = "manual_retrain"):
    print(f"\n================ AVVIO RETRAINING JOB [{run_id}] ================")
    dataset_repo = config["monitoring"]["monitoring_dataset_id"]
    token = os.environ.get("HF_TOKEN")

    # 1. Download buffer di accumulo da HF Dataset Hub
    print(f"[Retrain] Download buffer da {dataset_repo}/data/latest_retrain_buffer.parquet...")
    path = hf_hub_download(
        repo_id=dataset_repo,
        filename="data/latest_retrain_buffer.parquet",
        repo_type="dataset",
        token=token
    )
    buffer_df = pd.read_parquet(path)
    print(f"[Retrain] Record accumulati per il training: {len(buffer_df)}")

    # 2. Caricamento Golden Benchmark Test Set (immutabile)
    _, golden_test_df = load_raw_partitions()

    # 3. Addestramento del modello Challenger
    print("[Retrain] Addestramento Challenger in corso...")
    _, model, tokenizer = train_model(buffer_df, golden_test_df)
    challenger_dir = "./challenger"
    model.save_pretrained(challenger_dir)
    tokenizer.save_pretrained(challenger_dir)

    # 4. Valutazione Champion e Challenger sul Golden Benchmark
    chall_f1, champ_f1 = champion_challenger_check(challenger_dir, golden_test_df)

    # 5. Passaggio dati da GitHub Actions ad Airflow
    results = {
        "candidate_f1": round(float(chall_f1), 4),
        "champion_f1": round(float(champ_f1), 4),
        "run_id": run_id,
    }
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Retrain Job] Risultati generati: {results}")
    return results


if __name__ == "__main__":
    current_run_id = os.environ.get("RUN_ID", "local_run")
    run_retrain_job(run_id=current_run_id)
