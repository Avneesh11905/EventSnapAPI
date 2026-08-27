import pytest
import numpy as np
import cv2
import base64

from domain.entities import DetectedFace
from application.ports.face_services import IFaceDetector, IFaceEmbedder
from infrastructure.di_container import get_container


class MockFaceDetector(IFaceDetector):
    """Mocks the ONNX SCRFD Face Detector"""
    def detect_batch(
        self,
        images: list[np.ndarray],
        max_faces: int = 0,
        confidence: float | None = None,
        nms_threshold: float | None = None,
    ) -> list[list[DetectedFace]]:
        results = []
        for _ in images:
            # Fake a single face detection per image
            face = DetectedFace(
                bbox=np.array([10, 10, 50, 50], dtype=np.int32),
                landmarks=np.zeros((5, 2), dtype=np.float32),
                confidence=0.99,
            )
            results.append([face])
        return results


class MockFaceEmbedder(IFaceEmbedder):
    """Mocks the ONNX MobileFaceNet Embedder"""
    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        # Return a normalized dummy embedding vector of length 512 for each face
        embeddings = np.ones((len(aligned_faces), 512), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms


@pytest.fixture
def mock_container():
    """Provides a DI container with the heavy ML models mocked out."""
    container = get_container()
    with container.face_detector.override(MockFaceDetector()), \
         container.face_embedder.override(MockFaceEmbedder()):
        yield container


@pytest.fixture
def dummy_image_bgr() -> np.ndarray:
    """Returns a simple 100x100 green BGR image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 255, 0)
    return img


@pytest.fixture
def dummy_image_bytes(dummy_image_bgr: np.ndarray) -> bytes:
    """Returns the raw JPEG bytes of a dummy image."""
    _, buffer = cv2.imencode('.jpg', dummy_image_bgr)
    return buffer.tobytes()


@pytest.fixture
def dummy_image_base64(dummy_image_bytes: bytes) -> str:
    """Returns the base64 string of a dummy image."""
    return base64.b64encode(dummy_image_bytes).decode("utf-8")
