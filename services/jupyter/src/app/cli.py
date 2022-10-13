import typer
from app.config import settings

app = typer.Typer()


@app.command()
def train(
    model_ckpt: list[str] = typer.Option(
        ["distilbert-base-uncased"], help="List of Models in case of sweep."
    ),
    epochs : list[int] = typer.Option([1], help="List of number of epochs in case of sweep."),
    batch_size : list[int] = typer.Option([64], help="List of batch size"),
    sweep: bool = typer.Argument(False,help="If we want to run sweep")
):
    """Trains a model and pushes an artifact to weights and biases"""

    from app.model.train_model import train_routine
    train_routine(model_ckpt=model_ckpt,epochs=epochs,batch_size=batch_size,sweep=sweep)


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
