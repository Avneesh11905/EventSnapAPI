from typing import Protocol


class IImageAugmenter(Protocol):
    def augment(self, b64_images: list[str]) -> list[str]: ...
