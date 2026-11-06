import json
import os
from huggingface_hub import HfApi
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def run_deploy_job(model_folder: str = "./challenger", run_id: str = "manual_deploy"):
    """Carica la cartella con i pesi approvati sul Model Hub di Hugging Face promuovendola a Champion."""
    token = os.environ.get("HF_TOKEN")
    model_hub_id = os.environ.get("HF_HUB_ID", config["model"]["hub_id"])

    if not os.path.exists(model_folder):
        raise FileNotFoundError(
            f"Cartella artefatti '{model_folder}' non trovata. Impossibile eseguire il deploy."
        )

    if not token:
        raise ValueError("HF_TOKEN non impostato. Impossibile pubblicare sul Model Hub.")

    print(f"\n[Deploy] Inizio pubblicazione su Hugging Face Model Hub: {model_hub_id}")
    print(f"[Deploy] Run ID: {run_id}")

    api = HfApi()
    msg = f"deploy: promote challenger to champion from {run_id}"
    api.upload_folder(
        folder_path=model_folder,
        repo_id=model_hub_id,
        repo_type="model",
        token=token,
        commit_message=msg,
    )

    commit_sha = api.model_info(repo_id=model_hub_id, token=token).sha
    deploy_output = {"commit_sha": commit_sha, "run_id": run_id}
    with open("deploy_result.json", "w") as f:
        json.dump(deploy_output, f, indent=2)

    print(f"[Deploy Job] Nuovo modello pubblicato su HF! Commit SHA: {commit_sha}")
    return commit_sha


if __name__ == "__main__":
    current_run_id = os.environ.get("RUN_ID", "local_run")
    run_deploy_job(run_id=current_run_id)
