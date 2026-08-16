from dataclasses import dataclass
import numpy as np


@dataclass
class DetectedFace:
    """A single detected face with its bounding box and landmarks."""

    bbox: np.ndarray  # [x1, y1, x2, y2] in pixel coords
    landmarks: np.ndarray  # shape (5, 2)
    confidence: float
