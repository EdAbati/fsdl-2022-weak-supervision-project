import typer
from app.config import settings

app = typer.Typer()


@app.command()
def train(
    wandb_name: str = typer.Argument(
        "distilbert-base-uncased", help="Name for Weights and Biases Logging"
    ),
    epochs: int = typer.Argument(1, help="number of epochs"),
    batch_size: int = typer.Argument(64, help="batch size"),
):
    """Trains a model and pushes an artifact to weights and biases"""

    from app.model.train_model import train_routine

    train_routine(model_ckpt=wandb_name, epochs=epochs, batch_size=batch_size)


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
