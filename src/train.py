from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def train_model(train_df, test_df, push_to_hub=False):
    model_name = config['model']['base_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def tokenize_func(ex):
        return tokenizer(ex["text"], truncation=True)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_func, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_func, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        push_to_hub=push_to_hub,
        hub_model_id=config['model']['hub_id'] if push_to_hub else None,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator
    )

    trainer.train()

    return trainer, model, tokenizer
