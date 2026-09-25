from transformers import AutoTokenizer
import yaml


def test_tokenizer():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(cfg['model']['base_name'])
    assert tok("test")["input_ids"] is not None
