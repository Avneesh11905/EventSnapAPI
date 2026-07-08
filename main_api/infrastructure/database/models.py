from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Float
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid
import re

Base = declarative_base()

def sanitize_folder_path(path: str) -> str:
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', path)
    return clean.strip('_')

def get_event_table_name(folder_path: str) -> str:
    return f"event_{sanitize_folder_path(folder_path)}"

def create_event_table_model(folder_path: str):
    table_name = get_event_table_name(folder_path)
    
    if table_name in Base.metadata.tables:
        class EventModel(Base):
            __table__ = Base.metadata.tables[table_name]
        return EventModel
        
    class EventModel(Base):
        __tablename__ = table_name
        __table_args__ = {'extend_existing': True}
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        image_path = Column(String, index=True)
        embedding = Column(Vector(512))
        confidence = Column(Float)
        
    return EventModel
