from src.data_logic import get_smoke_test_data, load_raw_partitions
from src.train import train_model


def test_smoke():
    """Verifica rapida (25 record) del loop di training per la CI."""
    _, golden_test_df = load_raw_partitions()
    smoke_train = get_smoke_test_data(golden_test_df, n=25)
    smoke_eval = get_smoke_test_data(golden_test_df, n=10)

    trainer, model, tokenizer = train_model(smoke_train, smoke_eval)
    assert model is not None
    assert trainer.state.global_step > 0
