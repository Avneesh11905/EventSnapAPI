from .base import BaseConfig


class StorageSettings(BaseConfig):
    STORAGE_ENDPOINT: str
    STORAGE_ACCESS_KEY: str
    STORAGE_SECRET_KEY: str
    STORAGE_BUCKET_NAME: str
    S3_MAX_POOL_CONNECTIONS: int = 64


storage_settings = StorageSettings()
