from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MAIN_API_URL: str = "https://eventsnap-api.aymahajan.in"
    WEBHOOK_SECRET: str = "eventsnap_default_secret_please_change"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
