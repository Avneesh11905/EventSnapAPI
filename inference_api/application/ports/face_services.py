from typing import Protocol
import numpy as np
from domain.entities import DetectedFace


class IFaceDetector(Protocol):
    def detect_batch(
        self,
        images: list[np.ndarray],
        max_faces: int = 0,
        confidence: float | None = None,
        nms_threshold: float | None = None,
    ) -> list[list[DetectedFace]]: ...


class IFaceEmbedder(Protocol):
    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray: ...

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray: ...
