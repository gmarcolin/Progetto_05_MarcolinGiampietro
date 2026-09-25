from src.data_logic import apply_drift, get_stream_batches, load_raw_partitions


def test_stream_chunks_partitioning():
    """Verifica che il flusso venga diviso nel numero corretto di batch sequenziali."""
    stream_df, _ = load_raw_partitions()
    n_chunks = 5
    batches = get_stream_batches(stream_df, n_chunks=n_chunks)

    assert len(batches) == n_chunks
    assert all(len(b) > 0 for b in batches)
    # Verifica che la somma delle righe dei batch corrisponda al totale
    assert sum(len(b) for b in batches) == len(stream_df)


def test_drift_injection_removes_target_class():
    """Verifica che apply_drift abbatta drasticamente la presenza della classe positiva (label=2)."""
    stream_df, _ = load_raw_partitions()
    test_batch = stream_df.head(200)

    # Applichiamo un drift del 100% sulla classe 2 per testare la rimozione completa
    corrupted_batch = apply_drift(test_batch, target_class=2, ratio=1.0)

    assert 2 not in corrupted_batch["label"].values
