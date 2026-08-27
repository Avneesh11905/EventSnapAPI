from .base import BaseConfig


class QueueSettings(BaseConfig):
    RABBITMQ_URL: str = "amqp://guest:guest@es-rabbitmq:5672/"
    VALKEY_URL: str = "redis://localhost:6379/0"

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        url = self.VALKEY_URL
        if url.startswith("valkey://"):
            return url.replace("valkey://", "redis://", 1)
        if url.startswith("valkeys://"):
            return url.replace("valkeys://", "rediss://", 1)
        return url


queue_settings = QueueSettings()
