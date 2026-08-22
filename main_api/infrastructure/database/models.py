from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Float, Index, DateTime
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid


class Base(DeclarativeBase):
    pass


class EventEncodingModel(Base):
    __tablename__ = "event_encodings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid7)
    event_code = Column(String, index=True)
    image_path = Column(String, index=True)
    embedding = Column(Vector(512))
    confidence = Column(Float)

    __table_args__ = (
        Index(
            "ix_event_encodings_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class ProcessedImageModel(Base):
    """Tracks all images that have passed through the Celery encoding pipeline, including those with 0 faces."""

    __tablename__ = "processed_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid7)
    event_code = Column(String, index=True)
    image_path = Column(String, index=True)
    processed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        # Ensure we don't insert duplicate log entries if a task retries
        Index(
            "uix_processed_images_event_path", "event_code", "image_path", unique=True
        ),
    )
