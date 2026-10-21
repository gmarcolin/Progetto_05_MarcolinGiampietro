import os
from huggingface_hub import hf_hub_download
import pandas as pd
import yaml
from src.data_logic import load_raw_partitions
from src.evaluate import champion_challenger_check, evaluate_offline
from src.metrics_logger import log_batch_metrics
from src.train import train_model

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def run_retrain_pipeline(run_id: str = "run_manual"):
    print(f"\n================ AVVIO RETRAINING PIPELINE [{run_id}] ================")
    dataset_repo = config["monitoring"]["monitoring_dataset_id"]
    model_hub_id = os.environ.get("HF_HUB_ID", config["model"]["hub_id"])
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

    # 4. Sfida Champion vs Challenger sul Golden Benchmark
    is_better, challenger_f1, champion_f1 = champion_challenger_check(challenger_dir, golden_test_df)
    _, dist = evaluate_offline(challenger_dir, golden_test_df)

    # 5. Log delle metriche di validazione post-retrain su HF
    log_batch_metrics(
        batch_id=f"post_retrain_{run_id}",
        model_id=model_hub_id,
        f1_score=challenger_f1,
        distribution=dist,
        is_drifted=False,
        event_type="post_retrain_eval",
        run_id=run_id,
    )

    # 6. Scrittura output per GitHub Actions
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"deploy_approved={str(is_better).lower()}\n")
            f.write(f"challenger_f1={challenger_f1:.4f}\n")
            f.write(f"champion_f1={champion_f1:.4f}\n")

    decision_str = "APPROVATO" if is_better else "RESPINTO"
    print(
        f"[Gatekeeper] Esito: {decision_str} | Challenger F1: {challenger_f1:.4f} vs Champion F1: {champion_f1:.4f}"
    )
    return is_better, challenger_f1


if __name__ == "__main__":
    current_run_id = os.environ.get("RUN_ID", "local_run")
    run_retrain_pipeline(run_id=current_run_id)
