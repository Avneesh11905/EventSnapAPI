from application.ports.inference import IInferenceService
import httpx
from typing import List, Dict


class HFInferenceService(IInferenceService):
    def __init__(self, api_url: str, client: httpx.AsyncClient):
        self.api_url = api_url
        self.client = client

    async def get_face_encodings(
        self,
        b64_images: List[str],
        max_faces: int = 0,
        detection_conf: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> List[List[Dict]]:
        headers = {"Content-Type": "application/json"}
        payload = {
            "inputs": b64_images,
            "parameters": {
                "max_faces": max_faces,
                "detection_conf": detection_conf,
                "nms_threshold": nms_threshold,
            },
        }

        response = await self.client.post(
            f"{self.api_url}/", json=payload, headers=headers
        )
        response.raise_for_status()

        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Inference API Error: {data['error']}")

        return data.get("batch_faces", [])
