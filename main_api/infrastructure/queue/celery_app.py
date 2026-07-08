from celery import Celery
from config import settings

sync_db_backend = settings.DATABASE_URL.replace("postgresql+asyncpg://", "db+postgresql://")

celery_app = Celery(
    "eventsnap_tasks",
    broker=settings.RABBITMQ_URL,
    backend=sync_db_backend,
    include=['infrastructure.queue.celery_workers']
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
)
