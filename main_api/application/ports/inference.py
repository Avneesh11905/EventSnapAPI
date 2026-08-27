from typing import Protocol


class IInferenceService[T](Protocol):
    async def get_face_encodings(
        self,
        images: list[T],
        detection_conf: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> list[list[dict]]:
        """
        Returns a list of lists of face dictionaries.
        Outer list corresponds to the input images.
        Inner list contains the faces found in that image.
        Each face dict has 'embedding' and 'confidence'.
        """
        ...
