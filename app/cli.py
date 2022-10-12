import typer

from app.config import settings
from app.deploy import register_and_convert_model, register_artifact

app = typer.Typer()

app.command()(register_artifact)
app.command()(register_and_convert_model)


@app.command()
def train(
    model_checkpoint: str = typer.Option(
        "distilbert-base-uncased",
        help="Name of the HF pretrained model to fine tune.",
    ),
    epochs: int = typer.Option(1, help="Number of epochs"),
    batch_size: int = typer.Option(64, help="Batch size"),
):
    """Trains a model and pushes an artifact to weights and biases"""

    from app.model import train_routine

    train_routine(
        model_checkpoint=model_checkpoint, epochs=epochs, batch_size=batch_size
    )


@app.command()
def test(
    model_checkpoint: str = typer.Option(
        "distilbert-base-uncased",
        help="Name of the HF pretrained model to use for testing. The model must be regitered in the Hugging Face registry!",
    )
):
    """Test a model and print result to screen"""

    from app.model import test_routine

    test_routine(model_checkpoint=model_checkpoint)


@app.command()
def print_settings():
    color = typer.colors.BLUE
    bold = False

    for k, v in settings:
        if isinstance(v, list):
            typer.secho(f"{k}:", color=color, bold=bold)
            for e in v:
                typer.secho(f"\t{e}", color=color, bold=bold)
        else:
            typer.secho(f"{k}: {v}", color=color, bold=bold)


if __name__ == "__main__":
    app()
