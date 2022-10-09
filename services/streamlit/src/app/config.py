import enum
import os
import secrets
from typing import Optional

from pydantic import BaseSettings, HttpUrl, validator


class SettingsModeEnum(str, enum.Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class Settings(BaseSettings):
    DASHBOARD_MODE: SettingsModeEnum = SettingsModeEnum.DEV
    DASHBOARD_DEBUG: bool = False
    SECRET_KEY: str = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"

    PROJECT_NAME: str
    SENTRY_DSN: Optional[HttpUrl] = None

    @validator("SENTRY_DSN", pre=True)
    def sentry_dsn_can_be_blank(cls, v: str) -> Optional[str]:
        if bool(v) is False:
            return None
        return v


settings = Settings()