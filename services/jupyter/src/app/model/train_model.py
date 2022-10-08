from typing import Any, Optional

import boto3
import pandas as pd
import torch
import wandb
from datasets import ClassLabel, Dataset, Features, Value, load_dataset
from pydantic import BaseModel
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from app.config import settings


def load_data(dataset_name: str = "bergr7/weakly_supervised_ag_news") -> tuple:

    # files
    labeled_data_files = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }

    unlabeled_data_files = {"unlabeled": "unlabeled_train.csv"}
    # features
    labeled_features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(
                num_classes=4,
                names=["World", "Sports", "Business", "Sci/Tech"],
            ),
        }
    )
    unlabeled_features = Features({"text": Value("string")})

    # load data
    labeled_dataset_train = load_dataset(
        dataset_name,
        data_files=labeled_data_files,
        features=labeled_features,
        split="train",
    )

    labeled_dataset_val = load_dataset(
        dataset_name,
        data_files=labeled_data_files,
        features=labeled_features,
        split="validation",
    )

    labeled_dataset_test = load_dataset(
        dataset_name,
        data_files=labeled_data_files,
        features=labeled_features,
        split="test",
    )

    unlabeled_dataset = load_dataset(
        dataset_name,
        data_files=unlabeled_data_files,
        features=unlabeled_features,
    )

    return labeled_dataset_train, labeled_dataset_val, labeled_dataset_test


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

    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=_num_labels
    )

    # model.from_pretrained(model_ckpt, num_labels=_num_labels)

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

    return trainer


class WandbModelArtifact(BaseModel):

    entity: str
    project: str
    artifact_name: str
    tag: str

    def __str__(self):
        return f"{self.entity}/{self.project}/{self.artifact_name}:{self.tag}"


NUM_LABELS = 4


def upload_to_s3_bucket(
    torchscript_dir: str = "model.pt",
    bucket_name: str = "fsdl-model-test",
    s3_filename: str = "model.pt",
):
    s3 = boto3.client(
        "s3",
        settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    # upload file from local directory to s3 bucket
    with open(torchscript_dir, "rb") as f:
        s3.upload_fileobj(f, bucket_name, s3_filename)


def load_model_from_wandb(
    artifact_params: WandbModelArtifact, return_dir: bool = False
):

    run = wandb.init()
    artifact = run.use_artifact(str(artifact_params), type="model")
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
        tag="v0",
    )

    if model is None:
        load_model_from_wandb(w)

    # test model
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    test_encoded = test_dataset.map(tokenize, batched=True, batch_size=None)

    results = test_model(model, test_encoded, tokenizer)


def convert_model_to_torchscript(
    model_dir: str, model_ckpt: str
) -> torch.jit.ScriptModule:
    """Convert a saved model into TorchScript using tracing."""

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=NUM_LABELS, torchscript=True
    )

    # Create example input with
    # First row in 'ag_news' training dataset
    dummy_input = {
        "text": "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again.",
        "label": 2,
    }

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    dummy_tokenized_input = tokenizer(
        dummy_input["text"], truncation=True, return_tensors="pt"
    )

    return torch.jit.trace(model, tuple(dummy_tokenized_input.values()))


# %%
def convert_routine():
    model_ckpt = "distilbert-base-uncased"

    w = WandbModelArtifact(
        entity="team_44",
        project="model-registry",
        artifact_name="distilbert-base-uncased-finetuned-news",
        tag="v0",
    )

    model_dir = load_model_from_wandb(w, return_dir=True)

    out = convert_model_to_torchscript(
        model_dir=model_dir,
        model_ckpt=model_ckpt,
    )

    out.save("model.pt")

    return out


def train_routine():

    model_ckpt = "distilbert-base-uncased"

    model = get_model(
        # hidden_dp=0.1,
        # n_layers=4,
        model_ckpt=model_ckpt,
    )

    trainer = train_model(
        model,
        wandb_name=model_ckpt + "_test",
        model_ckpt=model_ckpt,
        epochs=1,
        batch_size=64,
    )


if __name__ == "__main__":
    # train_routine()
    # test_routine()
    convert_routine()
