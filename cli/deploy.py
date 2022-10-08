from pathlib import Path

import torch
import typer
import wandb
from rich import print
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODELS_PATH = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = LOCAL_MODELS_PATH / "traced_model.pt"

# Model Constants
DEFAULT_PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4

app = typer.Typer()


@app.command()
def register_artifact(artifact_name: str, model_name: str, model_alias: str = "prod") -> wandb.Artifact:
    """Register WandB artifact in the Model Registry"""
    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    is_linked = artifact.link(target_path=model_name, aliases=[model_alias])
    if is_linked:
        print(
            f"[green]Model [bold]'{artifact_name}'[/bold] successfully registered with name [bold]'{model_name}'[/bold] with model alias: [bold]'{model_alias}'[/bold][/green]"
        )
        model_registry_url = f"https://wandb.ai/{artifact.entity}/registry/model?selectionPath={model_name}"
        print(f"View at URL: [bold]{model_registry_url}[/bold]")
        return artifact


def convert_model_to_torchscript(model_dir: str) -> torch.jit.ScriptModule:
    """Convert a saved model into TorchScript using tracing."""
    print(f"Loading model from path '{model_dir}'")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=NUM_LABELS, torchscript=True)
    # First row in 'ag_news' training dataset
    dummy_input = {
        "text": "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again.",
        "label": 2,
    }
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PRETRAINED_MODEL_NAME)
    dummy_tokenized_input = tokenizer(dummy_input["text"], truncation=True, return_tensors="pt")
    print("Converting model to TorchScript")
    return torch.jit.trace(model, tuple(dummy_tokenized_input.values()))


@app.command()
def stage_model(artifact_name: str, model_name: str, model_alias: str = "prod"):
    model_artifact = register_artifact(artifact_name, model_name, model_alias)
    model_local_dir = model_artifact.download()
    traced_model = convert_model_to_torchscript(model_dir=model_local_dir)

    torch.jit.save(traced_model, DEFAULT_MODEL_PATH)
    return DEFAULT_MODEL_PATH


if __name__ == "__main__":
    app()
