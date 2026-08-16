from dataclasses import dataclass
from typing import List


@dataclass
class FaceEncoding:
    embedding: List[float]
    confidence: float


@dataclass
class MatchResult:
    image_path: str
    match_count: int
    best_distance: float


@dataclass
class AttendeeProfile:
    encodings: List[List[float]]


@dataclass
class Event:
    folder_path: str
