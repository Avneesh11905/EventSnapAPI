from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID

class FaceEncoding(BaseModel):
    embedding: List[float]
    confidence: float

class MatchResult(BaseModel):
    image_path: str
    match_count: int
    best_distance: float

class AttendeeProfile(BaseModel):
    encodings: List[List[float]]

class Event(BaseModel):
    folder_path: str
