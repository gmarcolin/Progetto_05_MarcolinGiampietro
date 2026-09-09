import os
import requests
import yaml
from src.data_logic import apply_imbalance, get_backtest_chunks, load_and_validate_data
from src.evaluate import get_model_f1
from src.metrics_logger import log_metric

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def run_monitoring_simulation():
    _, test_df = load_and_validate_data()
    chunks = get_backtest_chunks(test_df, n_chunks=config["data"]["backtest_chunks"])

    # 1. Valutazione su chunk normale (Finestra T-1)
    baseline_window = chunks[-2]
    f1_normal = get_model_f1(config["model"]["hub_id"], baseline_window)
    log_metric(
        batch_id="batch_pre_drift",
        model_id=config["model"]["hub_id"],
        f1_score=f1_normal,
        is_drifted=False,
        event_type="production_normal",
    )

    # 2. Simulazione Drift su finestra corrente (T)
    drifted_window = apply_imbalance(
        chunks[-1],
        target_class=config["monitoring"]["target_class_to_drop"],
        ratio=config["monitoring"]["imbalance_ratio"],
    )
    f1_drifted = get_model_f1(config["model"]["hub_id"], drifted_window)
    log_metric(
        batch_id="batch_drifted",
        model_id=config["model"]["hub_id"],
        f1_score=f1_drifted,
        is_drifted=True,
        event_type="drift_detected",
    )

    print(f"F1 Baseline: {f1_normal:.4f} -> F1 Drifted: {f1_drifted:.4f}")

    if f1_drifted < config["model"]["threshold_f1"]:
        print("Soglia violata. Triggering retraining workflow...")
        trigger_retraining()


def trigger_retraining():
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        print("GITHUB_TOKEN o GITHUB_REPOSITORY mancanti. Salto la chiamata API di dispatch.")
        return

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    requests.post(url, json={"event_type": "model_decay"}, headers=headers)


if __name__ == "__main__":
    run_monitoring_simulation()
