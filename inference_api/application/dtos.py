from dataclasses import dataclass


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
    batch_faces: list[list[FaceResultDTO]]
