import os
import yaml
from huggingface_hub import HfApi

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def deploy_champion_model(
    model_folder: str = "./challenger",
    run_id: str = None,
    f1_score: str = None,
):
    """Carica la cartella con i pesi approvati sul Model Hub di Hugging Face promuovendola a Champion."""
    token = os.environ.get("HF_TOKEN")
    model_hub_id = os.environ.get("HF_HUB_ID", config["model"]["hub_id"])
    run_id = run_id or os.environ.get("RUN_ID", "manual_run")
    f1_str = f1_score or os.environ.get("F1_SCORE", "N/A")

    if not os.path.exists(model_folder):
        raise FileNotFoundError(
            f"Cartella artefatti '{model_folder}' non trovata. Impossibile eseguire il deploy."
        )

    if not token:
        raise ValueError("HF_TOKEN non impostato. Impossibile pubblicare sul Model Hub.")

    print(f"\n[Deploy] Inizio pubblicazione su Hugging Face Model Hub: {model_hub_id}")
    print(f"[Deploy] Run ID: {run_id} | Golden Benchmark F1: {f1_str}")

    api = HfApi()
    commit_msg = (
        f"deploy: promote challenger to champion | run: {run_id} | Golden F1: {f1_str}"
    )

    api.upload_folder(
        folder_path=model_folder,
        repo_id=model_hub_id,
        repo_type="model",
        token=token,
        commit_message=commit_msg,
    )

    print(f"🎉 [Deploy] Nuovo Champion '{model_hub_id}' rilasciato con successo!\n")


if __name__ == "__main__":
    deploy_champion_model()
