from pydantic import field_validator

from .base import BaseConfig


class DatabaseSettings(BaseConfig):
    DATABASE_URL: str

    @field_validator("DATABASE_URL")
    @classmethod
    def set_asyncpg(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v


db_settings = DatabaseSettings()
