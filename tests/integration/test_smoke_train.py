from src.data_logic import load_and_validate_data
from src.train import train_model


def test_smoke():
    tr, te = load_and_validate_data(subset=5)
    trainer, model, _ = train_model(tr, te, push_to_hub=False)
    assert model is not None
