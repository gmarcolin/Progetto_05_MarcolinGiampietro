import os
import pandas as pd
from src.metrics_logger import METRICS_LOCAL, log_batch_metrics


def test_local_metric_logging():
    distribution = {"neg_pct": 10.0, "neu_pct": 20.0, "pos_pct": 70.0}
    log_batch_metrics("test_batch", 0.85, distribution, False, "unit_test")
    assert os.path.exists(METRICS_LOCAL)
    df = pd.read_csv(METRICS_LOCAL)
    assert not df.empty
    assert "f1_score" in df.columns
