from .base import BaseConfig


class InferenceSettings(BaseConfig):
    INFERENCE_API_URL: str
    INFERENCE_API_GRPC_URL: str | None = None
    INFERENCE_BATCH_SIZE: int = 64
    SIMILARITY_THRESHOLD: float = 0.45


inference_settings = InferenceSettings()
