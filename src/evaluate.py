import evaluate
import yaml
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def get_model_f1(model_path_or_id: str, test_df):
    """Esegue l'inferenza localmente in batch scaricando i pesi se necessario."""
    # Se per qualsiasi motivo l'input non è un DataFrame, forzalo
    if not isinstance(test_df, pd.DataFrame):
        test_df = pd.DataFrame(test_df)

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
        device=-1,  # CPU
        truncation=True,
        max_length=128,
    )

    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    texts = test_df["text"].astype(str).tolist()
    preds = pipe(texts, batch_size=16)

    y_pred = []
    for p in preds:
        lbl = p["label"].lower()
        if lbl in mapping:
            y_pred.append(mapping[lbl])
        elif lbl.startswith("label_"):
            y_pred.append(int(lbl.split("_")[-1]))
        else:
            y_pred.append(1)

    y_true = test_df["label"].tolist()
    f1_metric = evaluate.load("f1", trust_remote_code=True)
    return f1_metric.compute(predictions=y_pred, references=y_true, average="weighted")["f1"]


def champion_challenger_check(challenger_model_dir, test_df):
    """Confronta il Challenger locale con il Champion in produzione."""
    try:
        champion_f1 = get_model_f1(config["model"]["hub_id"], test_df)
    except Exception:
        champion_f1 = 0.0

    challenger_f1 = get_model_f1(challenger_model_dir, test_df)
    print(f"Champion F1: {champion_f1:.4f} | Challenger F1: {challenger_f1:.4f}")
    return challenger_f1 >= champion_f1, challenger_f1
