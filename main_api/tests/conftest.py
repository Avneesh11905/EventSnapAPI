import os

import pytest
import pytest_asyncio

# Set dummy environment variables for Pydantic Settings before importing any app modules
os.environ["DATABASE_URL"] = "postgresql://dummy:dummy@localhost:5432/dummy"
os.environ["STORAGE_ENDPOINT"] = "http://localhost:9000"
os.environ["STORAGE_ACCESS_KEY"] = "dummy"
os.environ["STORAGE_SECRET_KEY"] = "dummy"
os.environ["STORAGE_BUCKET_NAME"] = "dummy"
os.environ["INFERENCE_API_URL"] = "http://localhost:5000"

import alembic.command
import alembic.config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.rabbitmq import RabbitMqContainer
from testcontainers.community.redis import RedisContainer

from infrastructure.di_container import Container
from presentation.api.main import app


# Pytest Celery configurations
@pytest.fixture(scope="session")
def celery_config():
    return {
        "broker_url": os.environ.get("CELERY_BROKER_URL", "memory://"),
        "result_backend": os.environ.get("CELERY_RESULT_BACKEND", "cache+memory://"),
    }


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg18-trixie") as postgres:
        # Generate the asyncpg URL
        url = postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql+asyncpg")

        # Override db_settings globally so Alembic and other tools use the live container
        from config.database import db_settings

        db_settings.DATABASE_URL = url

        # Run Alembic migrations
        alembic_cfg = alembic.config.Config("alembic.ini")
        alembic_cfg.set_main_option(
            "sqlalchemy.url", url.replace("+asyncpg", "")
        )  # Alembic sync driver
        alembic.command.upgrade(alembic_cfg, "head")

        yield url


@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer("redis:8.8.0") as redis:
        yield f"redis://{redis.get_container_host_ip()}:{redis.get_exposed_port(6379)}/0"


@pytest.fixture(scope="session")
def rabbitmq_container():
    with RabbitMqContainer("rabbitmq:3-management") as rabbitmq:
        host = rabbitmq.get_container_host_ip()
        port = rabbitmq.get_exposed_port(5672)
        yield f"amqp://guest:guest@{host}:{port}/"


@pytest_asyncio.fixture(scope="function")
async def db_engine(postgres_container):
    engine = create_async_engine(postgres_container, pool_pre_ping=True)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine):
    async_session = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def app_container(postgres_container, redis_container, rabbitmq_container):
    container = Container()

    # Override configuration to point to Testcontainers
    container.config.from_dict(
        {
            "db_url": postgres_container,
            "storage_endpoint": "http://localhost:9000",
            "storage_bucket": "test-bucket",
            "storage_access": "test",
            "storage_secret": "test",
            "inference_url": "http://localhost:5000",
        }
    )

    yield container
    container.unwire()


@pytest_asyncio.fixture(scope="function")
async def test_client(app_container):
    # Ensure the app uses our test container
    app.container = app_container

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
