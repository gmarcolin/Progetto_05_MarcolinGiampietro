import os
import pandas as pd
from src.metrics_logger import METRICS_LOCAL, log_batch_metrics


def test_local_metric_logging():
    dist = {"neg_pct": 10.0, "neu_pct": 20.0, "pos_pct": 70.0}

    log_batch_metrics(
        batch_id="test_batch",
        model_id="test_model_v1",
        f1_score=0.85,
        distribution=dist,
        is_drifted=False,
        event_type="unit_test"
    )

    assert os.path.exists(METRICS_LOCAL)
    df = pd.read_csv(METRICS_LOCAL)
    assert not df.empty
    assert "f1_score" in df.columns
    assert df["model_id"].iloc[-1] == "test_model_v1"
