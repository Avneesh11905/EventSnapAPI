from typing import Protocol, TypeVar
import numpy as np

T = TypeVar("T", str, bytes)

class IImageDecoder(Protocol[T]):
    def decode(self, input_data: T) -> np.ndarray: ...
    def decode_batch(self, inputs: list[T]) -> list[np.ndarray]: ...
