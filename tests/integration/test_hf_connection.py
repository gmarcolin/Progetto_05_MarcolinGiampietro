import os
from huggingface_hub import HfApi


def test_api():
    api = HfApi()
    assert api.whoami(token=os.getenv("HF_TOKEN")) is not None
