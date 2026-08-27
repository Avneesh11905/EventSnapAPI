from celery import Celery
from config.queue import queue_settings

celery_app = Celery(
    "eventsnap_tasks",
    broker=queue_settings.RABBITMQ_URL,
    backend=queue_settings.CELERY_RESULT_BACKEND,
    include=["infrastructure.queue.celery_workers"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    result_expires=3600,
)
