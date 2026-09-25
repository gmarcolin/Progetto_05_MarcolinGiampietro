import pandas as pd
import yaml
from src.evaluate import evaluate_batch

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def test_model_inference_and_distribution():
    """Verifica che la funzione di inferenza calcoli correttamente metriche e distribuzioni di sentiment."""
    # Mini-dataset con esempi chiari per le tre classi
    sample_df = pd.DataFrame(
        {
            "text": [
                "I absolutely love this new MLOps pipeline, it is wonderful!",  # Positivo (2)
                "This is the worst flight delay ever, terrible service.",  # Negativo (0)
                "The train arrives at 10:30 am tomorrow morning.",  # Neutro (1)
            ],
            "label": [2, 0, 1],
        }
    )

    # Testiamo l'inferenza usando il modello base configurato
    model_to_test = config["model"]["base_name"]
    f1_score, distribution = evaluate_batch(model_to_test, sample_df)

    # Asserzioni su integrità dell'output
    assert isinstance(f1_score, float)
    assert 0.0 <= f1_score <= 1.0

    # Verifica che il dizionario della distribuzione contenga le chiavi attese
    assert "neg_pct" in distribution
    assert "neu_pct" in distribution
    assert "pos_pct" in distribution

    # La somma delle percentuali deve fare 100%
    total_pct = distribution["neg_pct"] + distribution["neu_pct"] + distribution["pos_pct"]
    assert round(total_pct) == 100
