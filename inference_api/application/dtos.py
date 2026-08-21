from dataclasses import dataclass
from typing import List

@dataclass
class InferenceParametersDTO:
    max_faces: int
    detection_conf: float
    nms_threshold: float

@dataclass
class FaceResultDTO:
    bbox: list[float]
    confidence: float
    embedding: list[float]

@dataclass
class InferenceResultDTO:
    batch_faces: List[List[FaceResultDTO]]
