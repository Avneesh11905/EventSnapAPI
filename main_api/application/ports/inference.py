from typing import List, Dict, Protocol

class InferenceService(Protocol):
    async def get_face_encodings(
        self, 
        b64_images: List[str], 
        max_faces: int = 0, 
        detection_conf: float = 0.5, 
        nms_threshold: float = 0.4
    ) -> List[List[Dict]]:
        """
        Returns a list of lists of face dictionaries.
        Outer list corresponds to the input b64_images.
        Inner list contains the faces found in that image.
        Each face dict has 'embedding' and 'confidence'.
        """
        pass
