from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):

    HF_TOKEN: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None
    AWS_REGION: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None


settings = Settings()

NUM_LABELS = 4
