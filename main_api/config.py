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
    INFERENCE_API_GRPC_URL: str | None = None

    VALKEY_URL: str = "redis://localhost:6379/0"

    SIMILARITY_THRESHOLD: float = 0.45
    INFERENCE_BATCH_SIZE: int = 64
    S3_MAX_POOL_CONNECTIONS: int = 64

    CORS_ORIGINS: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
