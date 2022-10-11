from typing import Any, Optional

import wandb
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from app.config import NUM_LABELS, settings
from app.data import load_data


def get_model(model_ckpt: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=NUM_LABELS
    )
    return model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def test_model(model, test_data, tokenizer):
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    model.eval()
    trainer.predict(test_data)
    return trainer


def train_model(
    model,
    wandb_name: str,
    model_ckpt: str,
    epochs: int = 1,
    batch_size: int = 64,
):
    train_dataset, val_dataset, _ = load_data()

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
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=True,
        report_to="wandb",
        run_name=wandb_name,
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

    return trainer


class WandbModelArtifact(BaseModel):

    entity: str
    project: str
    artifact_name: str
    tag: str

    def __str__(self):
        return f"{self.entity}/{self.project}/{self.artifact_name}:{self.tag}"


def load_model_from_wandb(
    artifact_params: WandbModelArtifact, return_dir: bool = False
):

    api = wandb.Api()
    artifact = api.artifact(str(artifact_params), type="model")
    artifact_dir = artifact.download()

    if return_dir:
        return artifact_dir

    model = AutoModelForSequenceClassification.from_pretrained(
        artifact_dir,
        num_labels=NUM_LABELS,
        torchscript=True,
    )

    return model


def test_routine(model: Optional[Any] = None):

    model_ckpt = "distilbert-base-uncased"

    _, _, test_dataset = load_data()

    w = WandbModelArtifact(
        entity="team_44",
        project="model-registry",
        artifact_name="distilbert-base-uncased-finetuned-news",
        tag="prod",
    )

    if model is None:
        load_model_from_wandb(w)

    # test model
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    test_encoded = test_dataset.map(tokenize, batched=True, batch_size=None)

    results = test_model(model, test_encoded, tokenizer)


def train_routine(
    model_ckpt: str = "distilbert-base-uncased",
    epochs: int = 1,
    batch_size: int = 64,
):

    model = get_model(
        model_ckpt=model_ckpt,
    )

    train_model(
        model,
        wandb_name=model_ckpt + "_test",
        model_ckpt=model_ckpt,
        epochs=epochs,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    train_routine()
