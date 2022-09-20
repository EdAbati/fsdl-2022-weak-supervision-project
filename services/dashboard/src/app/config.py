import enum
import os
import secrets
from typing import Optional

from pydantic import AnyHttpUrl, BaseSettings, HttpUrl, validator


class SettingsModeEnum(str, enum.Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class Settings(BaseSettings):
    DASHBOARD_MODE: SettingsModeEnum = SettingsModeEnum.DEV
    DASHBOARD_DEBUG: bool = False
    SECRET_KEY: str = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    SERVER_NAME: str
    SERVER_HOST: AnyHttpUrl

    PROJECT_NAME: str
    SENTRY_DSN: Optional[HttpUrl] = None

    @validator("SENTRY_DSN", pre=True)
    def sentry_dsn_can_be_blank(cls, v: str) -> Optional[str]:
        if bool(v) is False:
            return None
        return v

    BACKEND_CORS_ORIGINS: Optional[list[str]] = None

    API_DOCKER_TAG: str = "latest"


settings = Settings()
