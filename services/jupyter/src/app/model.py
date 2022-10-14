import wandb
from pydantic import BaseModel
from rich import print
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from app.config import NUM_LABELS, settings
from app.data import load_data

DEFAULT_WANDB_ENTITY = "team_44"


def get_model(model_checkpoint: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=NUM_LABELS
    )
    return model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def train_model(
    config=None
):
    with wandb.init(project="huggingface", entity=DEFAULT_WANDB_ENTITY,config=config) as run:
        # set sweep configuration
        config = wandb.config
        batch_size = config.batch_size
        epochs = config.epochs
        model_checkpoint = config.model_checkpoint

    datasets = load_data()

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized_datasets = datasets.map(tokenize, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns("text")
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer)

    logging_steps = len(tokenized_datasets["train"]) // batch_size
    model_name = f"{model_checkpoint}-finetuned-news"

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
        log_level="error",
        hub_token=settings.HF_TOKEN,
    )
    trainer = Trainer(
        model=get_model(model_checkpoint),
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
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


def test_model(model, test_data, tokenizer):
    args = TrainingArguments(output_dir="tmp_test_model", report_to="none")
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer.evaluate(test_data)


def test_routine(
    model_checkpoint: str = "distilbert-base-uncased",
) -> None:


    test_dataset = load_data(split="test")

    artifact_details = WandbModelArtifact(
        entity=DEFAULT_WANDB_ENTITY,
        project="model-registry",
        artifact_name=f"{model_checkpoint}-finetuned-news",
        tag="prod",
    )
    model = load_model_from_wandb(artifact_details)

    # test model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    test_tokenized = test_dataset.map(tokenize, batched=True, batch_size=None)
    test_tokenized = test_tokenized.remove_columns("text")
    test_tokenized.set_format("torch")

    results = test_model(model, test_tokenized, tokenizer)
    print("[bold]Test Evaluation Metrics:[/bold]")
    print(results)


def do_sweep(model_checkpoint,batch_size,epochs):
    sweep_params = {}
    sweep_available_params = {
        "model_checkpoint":{"values":model_checkpoint},
        "epochs": {"values": epochs},
        'batch_size': {'values': batch_size},
    }
    
    params = list(sweep_available_params.keys())
    
    for p in params:
        sweep_params[p] = sweep_available_params[p]
         
    sweep_config = {
        "name": "test-sweep",
        "method": "grid",  # random, #grid
        "parameters": sweep_params,
    }
    sweep_id = wandb.sweep(sweep_config,entity="team_44",project="huggingface")
    wandb.agent(sweep_id, train_model)

def train_routine(
    model_checkpoint:list[str] =["distilbert-base-uncased"],
    epochs:list[int]=[1],
    batch_size:list[int]=[64],
    sweep:bool = False
) -> None:
    
    if sweep:
        do_sweep(model_checkpoint,batch_size,epochs)
    else:
        default_config = {
        "model_checkpoint":model_checkpoint[0],
        "batch_size": batch_size[0],
        "epochs":epochs[0],
        }
        train_model(config=default_config)

if __name__ == "__main__":
    train_routine()
