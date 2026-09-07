from src.data_logic import load_and_validate_data, apply_imbalance


def test_imbalance():
    _, te = load_and_validate_data(subset=20)
    corrupted = apply_imbalance(te, ratio=1.0, target_class=2)
    assert 2 not in corrupted['label'].values
