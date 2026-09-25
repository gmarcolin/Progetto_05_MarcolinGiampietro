from datasets import Dataset
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def train_model(train_buffer_df: pd.DataFrame, eval_df: pd.DataFrame):
    """Addestra il Challenger sul buffer di accumulo dei dati di produzione."""
    model_name = config["model"]["base_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Subsample per non superare le risorse CPU del runner
    if len(train_buffer_df) > 1500:
        train_buffer_df = train_buffer_df.sample(n=1500, random_state=42)

    def tokenize_func(ex):
        return tokenizer(ex["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_buffer_df).map(tokenize_func, batched=True)
    eval_ds = Dataset.from_pandas(eval_df.head(200)).map(tokenize_func, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        save_strategy="no",
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        weight_decay=0.01,
        push_to_hub=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    return trainer, model, tokenizer
