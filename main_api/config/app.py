from .base import BaseConfig


class AppSettings(BaseConfig):
    CORS_ORIGINS: str = "http://localhost:3000"


app_settings = AppSettings()
