import os
import pytest
from huggingface_hub import HfApi


def test_hf_hub_write_permission():
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN non configurato")
    api = HfApi()
    user = api.whoami(token=token)
    assert "name" in user
