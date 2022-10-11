from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):

    PROJECT_NAME: str
    HF_TOKEN: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()

NUM_LABELS = 4
