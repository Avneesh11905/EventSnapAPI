from celery import Celery
from config import settings

# Convert asyncpg URL to standard psycopg2 URL for Celery's sync worker
# 'postgresql+asyncpg://...' -> 'db+postgresql://...'
sync_db_backend = settings.DATABASE_URL.replace("postgresql+asyncpg://", "db+postgresql://")

# Initialize Celery with RabbitMQ as Broker and PostgreSQL as Results Backend
celery_app = Celery(
    "eventsnap_tasks",
    broker=settings.RABBITMQ_URL,
    backend=sync_db_backend,
    include=['tasks']
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Worker optimizations
    worker_prefetch_multiplier=1,  # Good for heavy tasks like querying MinIO
    task_acks_late=True,           # Retry if worker crashes mid-task
)
