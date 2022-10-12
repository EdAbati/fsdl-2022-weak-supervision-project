from typing import Optional

from pydantic import BaseSettings
from dotenv import load_dotenv

class Settings(BaseSettings):

    PROJECT_NAME: str
    HF_TOKEN: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None

load_dotenv()
settings = Settings()

NUM_LABELS = 4
