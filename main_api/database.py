from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from config import settings

from sqlalchemy.pool import NullPool

# Create PostgreSQL Engine natively async
engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)

# Create an async session factory
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

# Base class for declarative models
Base = declarative_base()

async def get_db():
    """Dependency injection to get DB session per FastAPI request."""
    async with SessionLocal() as db:
        yield db

async def init_db():
    """Ensure pgvector extension exists."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
