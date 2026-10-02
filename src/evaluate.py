import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import f1_score
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def evaluate_batch(model_path_or_id: str, batch_df: pd.DataFrame):
    """Valuta un batch calcolando F1-score e la distribuzione percentuale del sentiment predetto."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_path_or_id)
    except Exception:
        # Fallback al modello base se l'ID custom non è ancora presente
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_name"])
        model = AutoModelForSequenceClassification.from_pretrained(config["model"]["base_name"])

    pipe = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        # device=-1,  # CPU
        # truncation=True,
        # max_length=128,
    )

    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    texts = batch_df["text"].astype(str).tolist()
    preds = pipe(texts, batch_size=16)

    y_pred = []
    for p in preds:
        lbl = p["label"].lower()
        if lbl in mapping:
            y_pred.append(mapping[lbl])
        # elif lbl.startswith("label_"):
        #     y_pred.append(int(lbl.split("_")[-1]))
        else:
            raise ValueError(f"Unexpected model label: {p['label']}")
            # y_pred.append(1)

    y_true = batch_df["label"].tolist()

    # Calcolo F1 pesato
    # f1_metric = evaluate.load("f1", trust_remote_code=True)
    # f1_score = f1_metric.compute(predictions=y_pred, references=y_true, average="weighted")["f1"]
    f1 = f1_score(y_true, y_pred, average="macro")

    # Calcolo distribuzione del sentiment predetto (%)
    total = len(y_pred)
    neg_pct = round(float(np.sum(np.array(y_pred) == 0) / total * 100), 2)
    neu_pct = round(float(np.sum(np.array(y_pred) == 1) / total * 100), 2)
    pos_pct = round(float(np.sum(np.array(y_pred) == 2) / total * 100), 2)

    distribution = {"neg_pct": neg_pct, "neu_pct": neu_pct, "pos_pct": pos_pct}

    return f1, distribution


def champion_challenger_check(challenger_dir: str, golden_test_df: pd.DataFrame):
    """Confronta Challenger e Champion esclusivamente sul Golden Benchmark Test Set."""
    try:
        champion_f1, _ = evaluate_batch(config["model"]["hub_id"], golden_test_df)
    except Exception:
        champion_f1, _ = evaluate_batch(config["model"]["base_name"], golden_test_df)

    challenger_f1, _ = evaluate_batch(challenger_dir, golden_test_df)
    print(f"[Gatekeeper] Champion F1: {champion_f1:.4f} | Challenger F1: {challenger_f1:.4f}")
    return challenger_f1 >= champion_f1, challenger_f1, champion_f1
