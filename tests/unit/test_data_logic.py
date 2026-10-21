from src.data_logic import apply_drift, get_batch_by_date, load_raw_partitions


def test_partitions_and_dates():
    """Verifica che il dataset venga diviso correttamente."""
    stream_df, golden_test_df = load_raw_partitions()
    assert "event_date" in stream_df.columns
    assert len(stream_df) > 0
    assert len(golden_test_df) > 0

    first_date = stream_df["event_date"].iloc[0]
    batch = get_batch_by_date(stream_df, first_date)
    assert not batch.empty
    assert (batch["event_date"] == first_date).all()


def test_drift_removal():
    """Verifica che apply_drift abbatta drasticamente la presenza della classe positiva (label=2)."""
    stream_df, _ = load_raw_partitions()
    batch = stream_df.head(100)

    # Applichiamo un drift del 100% sulla classe 2 per testare la rimozione completa
    drifted = apply_drift(batch, target_class=2, ratio=1.0)

    assert 2 not in drifted["label"].values
