import typer
from myapp.config import settings, SettingsModeEnum

app = typer.Typer()

if __name__ == "__main__":
    app()


@app.command()
def print_settings():
    is_prod = settings.FASTAPI_MODE == SettingsModeEnum.PROD
    color = typer.colors.RED if is_prod else typer.colors.BLUE
    bold = is_prod

    for k, v in settings:
        if isinstance(v, list):
            typer.secho(f"{k}:", color=color, bold=bold)
            for e in v:
                typer.secho(f"\t{e}", color=color, bold=bold)
        else:
            typer.secho(f"{k}: {v}", color=color, bold=bold)
