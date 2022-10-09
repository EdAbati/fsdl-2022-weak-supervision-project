import enum
from typing import Optional

from pydantic import BaseSettings, HttpUrl, validator


class SettingsModeEnum(str, enum.Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class Settings(BaseSettings):

    PROJECT_NAME: str
    SERVICE_MODE: SettingsModeEnum = SettingsModeEnum.PROD
    HF_TOKEN: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None


settings = Settings()
