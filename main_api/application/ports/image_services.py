from typing import List, Protocol


class IImageAugmenter(Protocol):
    def augment(self, b64_images: List[str]) -> List[str]: ...
