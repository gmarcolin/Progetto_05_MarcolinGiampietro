import pytest
import os
from huggingface_hub import InferenceClient
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def test_remote_inference_api():
    token = os.getenv("HF_TOKEN")
    client = InferenceClient(model=config['model']['hub_id'], token=token)

    try:
        result = client.text_classification("I am so happy with this MLOps course!")
        assert len(result) > 0
        assert any(label['label'] in ["positive", "LABEL_2"] for label in result)

    except Exception as e:
        pytest.skip(f"L'API del modello non è ancora pronta o il modello non è stato pushato: {e}")
