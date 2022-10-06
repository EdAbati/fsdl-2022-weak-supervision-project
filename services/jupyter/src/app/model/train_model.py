from typing import Optional

import pandas as pd
from app.config import settings
from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def load_data(dataset_uri: str = "bergr7/weakly_supervised_ag_news") -> tuple:

    train_dataset = load_dataset(dataset_uri, split="train")
    val_dataset = load_dataset(dataset_uri, split="validation")

    return train_dataset, val_dataset


def get_model(
    # hidden_dp: float,
    # n_layers: int,
    model_ckpt: str,
):

    _num_labels: int = 4

    # config = AutoConfig.from_pretrained(model_ckpt, num_labels=_num_labels)
    # config.hidden_dropout_prob = hidden_dp
    # config.attention_probs_dropout_prob = 0.1
    # config.n_layers = n_layers

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt,num_labels=_num_labels)

    #model.from_pretrained(model_ckpt, num_labels=_num_labels)

    return model


# %%
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


from transformers import Trainer, TrainingArguments


def train_model(
    model,
    wandb_name: str,
    model_ckpt: str,
    epochs: int = 1,
    batch_size: int = 64,
):

    train_dataset, val_dataset = load_data()

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    train_encoded = train_dataset.map(tokenize, batched=True, batch_size=None)
    val_encoded = val_dataset.map(tokenize, batched=True, batch_size=None)

    logging_steps = len(train_encoded) // batch_size
    model_name = f"{model_ckpt}-finetuned-news"

    training_args = TrainingArguments(
        output_dir=model_name,
        num_train_epochs=epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        # optim="adamw_torch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=True,
        report_to="wandb",
        run_name=wandb_name,  # "distill-bert-base-config",
        log_level="error",
        hub_token=settings.HF_TOKEN,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        tokenizer=tokenizer,
    )

    trainer.train()


# %%


if __name__ == "__main__":

    model_ckpt = "distilbert-base-uncased"

    model = get_model(
        # hidden_dp=0.1,
        # n_layers=4,
        model_ckpt=model_ckpt,
    )

    train_model(
        model,
        wandb_name=model_ckpt + "_test",
        model_ckpt=model_ckpt,
        epochs=1,
        batch_size=64,
    )
