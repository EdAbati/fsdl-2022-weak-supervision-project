from typing import Optional
import typer

from app.config import SettingsModeEnum, settings

app = typer.Typer()


@app.command()
def train(
    p1: float = typer.Argument(..., help="hidden dropout probability"),
    p2: Optional[float] = typer.Argument(None, help="hidden dropout probability"),
    wandb_name: str = typer.Argument(..., help="Name for Weights and Biases Logging"),
    num_layers: int = typer.Argument(..., help="number of layers in the model"),
):
    pass


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
