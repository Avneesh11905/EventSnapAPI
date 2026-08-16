from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Float
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
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

