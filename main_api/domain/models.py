from dataclasses import dataclass


@dataclass
class FaceEncoding:
    embedding: list[float]
    confidence: float


@dataclass
class MatchResult:
    image_path: str
    match_count: int
    best_distance: float


@dataclass
class Event:
    folder_path: str
