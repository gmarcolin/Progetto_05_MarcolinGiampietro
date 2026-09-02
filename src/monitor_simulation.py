import os, requests, yaml
from src.data_logic import load_and_validate_data, get_backtest_chunks, apply_imbalance
from src.evaluate import get_model_f1

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def run_monitoring_simulation():
    _, test_df = load_and_validate_data()
    chunks = get_backtest_chunks(test_df, n_chunks=config['data']['backtest_chunks'])
    
    # Simulazione Drift: Sbilanciamento sull'ultimo chunk
    production_window = apply_imbalance(chunks[-1], ratio=config['monitoring']['imbalance_ratio'])
    
    current_f1 = get_model_f1(config['model']['hub_id'], production_window)
    print(f"Monitoraggio: F1 calcolata {current_f1}")

    if current_f1 < config['model']['threshold_f1']:
        print("Model Decay rilevato! Inviando trigger...")
        trigger_retraining()

def trigger_retraining():
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    url = f"https://api.github.com/repos/{repo}/dispatches"
    requests.post(url, json={"event_type": "model_decay"}, 
                  headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"})

if __name__ == "__main__":
    run_monitoring_simulation()