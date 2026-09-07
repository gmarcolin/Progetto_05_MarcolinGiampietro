import evaluate
from transformers import pipeline
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def get_model_f1(model_path, test_df):
    pipe = pipeline("sentiment-analysis", model=model_path, device=-1)
    mapping = {"negative": 0, "neutral": 1, "positive": 2}

    preds = pipe(test_df['text'].tolist(), truncation=True)
    # Gestione diverse nomenclature label
    y_pred = [mapping[p['label']] if p['label'] in mapping else int(p['label'].split('_')[-1]) for p in preds]
    y_true = test_df['label'].tolist()

    f1_metric = evaluate.load("f1")
    return f1_metric.compute(predictions=y_pred, references=y_true, average="weighted")['f1']


def champion_challenger_check(challenger_model_dir, test_df):
    try:
        champion_f1 = get_model_f1(config['model']['hub_id'], test_df)
    except Exception:
        champion_f1 = 0.0

    challenger_f1 = get_model_f1(challenger_model_dir, test_df)
    return challenger_f1 > champion_f1, challenger_f1
