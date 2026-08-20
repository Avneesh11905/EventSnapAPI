from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str

    @field_validator("DATABASE_URL")
    @classmethod
    def set_asyncpg(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    RABBITMQ_URL: str = "amqp://guest:guest@es-rabbitmq:5672/"

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        # Celery requires a synchronous driver (psycopg2) for the result backend
        sync_url = self.DATABASE_URL.replace("+asyncpg", "")
        # psycopg2 expects sslmode=require, whereas asyncpg expects ssl=require
        sync_url = sync_url.replace("ssl=require", "sslmode=require")
        return f"db+{sync_url}"

    STORAGE_ENDPOINT: str
    STORAGE_ACCESS_KEY: str
    STORAGE_SECRET_KEY: str
    STORAGE_BUCKET_NAME: str

    INFERENCE_API_URL: str

    SIMILARITY_THRESHOLD: float = 0.55
    MIN_MATCHES: int = 2

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
