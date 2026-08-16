from typing import Protocol
import numpy as np


class IImageDecoder(Protocol):
    def decode(self, b64_str: str) -> np.ndarray: ...
    def decode_batch(self, b64_strings: list[str]) -> list[np.ndarray]: ...
