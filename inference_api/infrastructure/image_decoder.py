import base64
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from application.ports.image_services import IImageDecoder
from domain.exceptions import InvalidInputFormatError

logger = logging.getLogger(__name__)


class Base64ImageDecoder(IImageDecoder[str]):
    """
    Decodes Base64 strings into OpenCV-compatible NumPy arrays (BGR).
    Uses cv2.imdecode for maximum speed.
    Used by the HTTP presentation layer.
    """

    def decode(self, input_data: str) -> np.ndarray:
        if not isinstance(input_data, str):
            raise InvalidInputFormatError("Invalid input format.")

        if "," in input_data:
            input_data = input_data.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(input_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_img is None:
                raise ValueError("cv2.imdecode failed to parse image bytes")

            return cv_img

        except Exception as e:
            raise InvalidInputFormatError(f"Failed to decode base64 image: {e}")

    def decode_batch(self, inputs: list[str]) -> list[np.ndarray]:
        with ThreadPoolExecutor() as pool:
            return list(pool.map(self.decode, inputs))


class BytesImageDecoder(IImageDecoder[bytes]):
    """
    Decodes raw image bytes into OpenCV-compatible NumPy arrays (BGR).
    Used by the gRPC presentation layer — eliminates base64 encode/decode
    overhead by accepting native binary data directly from protobuf.
    """

    def decode(self, input_data: bytes) -> np.ndarray:
        if not isinstance(input_data, (bytes, bytearray)):
            raise InvalidInputFormatError(f"Expected bytes, got {type(input_data).__name__}")
        try:
            np_arr = np.frombuffer(input_data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ValueError("cv2.imdecode failed — unsupported or corrupt image bytes")
            return cv_img
        except Exception as e:
            raise InvalidInputFormatError(f"Failed to decode image bytes: {e}")

    def decode_batch(self, inputs: list[bytes]) -> list[np.ndarray]:
        with ThreadPoolExecutor() as pool:
            return list(pool.map(self.decode, inputs))
