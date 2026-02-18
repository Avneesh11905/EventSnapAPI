from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    API_KEY: str 

    DATABASE_URL: str
    
    RABBITMQ_URL: str
    
    MINIO_ENDPOINT: str 
    MINIO_ACCESS_KEY: str 
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str 
    
    INFERENCE_API_URL: str 
    INFERENCE_API_TOKEN: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
