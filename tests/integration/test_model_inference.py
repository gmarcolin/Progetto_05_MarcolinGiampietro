from transformers import pipeline
import yaml
def test_inf():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    p = pipeline("sentiment-analysis", model=cfg['model']['base_name'])
    assert p("I love AI") is not None