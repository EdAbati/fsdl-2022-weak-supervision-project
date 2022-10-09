import enum
from typing import Optional

from pydantic import BaseSettings, HttpUrl, validator


class SettingsModeEnum(str, enum.Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class Settings(BaseSettings):

    PROJECT_NAME: str
    SENTRY_DSN: Optional[HttpUrl] = None

    @validator("SENTRY_DSN", pre=True)
    def sentry_dsn_can_be_blank(cls, v: str) -> Optional[str]:
        if bool(v) is False:
            return None
        return v

    HF_TOKEN: Optional[str] = None
    WANDB_API_KEY: Optional[str] = None


settings = Settings()
