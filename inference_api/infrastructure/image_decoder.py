import base64
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from domain.exceptions import InvalidInputFormatError
from application.ports.image_services import IImageDecoder


class Base64ImageDecoder(IImageDecoder):
    """
    Decodes Base64 strings into OpenCV-compatible NumPy arrays (BGR).
    Uses cv2.imdecode for maximum speed.
    """

    def decode(self, b64_str: str) -> np.ndarray:
        if not isinstance(b64_str, str):
            raise InvalidInputFormatError("Invalid input format.")

        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(b64_str)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_img is None:
                raise ValueError("cv2.imdecode failed to parse image bytes")

            return cv_img

        except Exception as e:
            raise InvalidInputFormatError(f"Failed to decode base64 image: {e}")

    def decode_batch(self, b64_strings: list[str]) -> list[np.ndarray]:
        with ThreadPoolExecutor() as pool:
            return list(pool.map(self.decode, b64_strings))
