from application.ports.inference import IInferenceService
import httpx
from typing import List, Dict


class HFInferenceService(IInferenceService):
    def __init__(self, api_url: str):
        self.api_url = api_url

    async def get_face_encodings(
        self,
        b64_images: List[str],
        detection_conf: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> List[List[Dict]]:
        headers = {"Content-Type": "application/json"}
        payload = {
            "inputs": b64_images,
            "parameters": {
                "detection_conf": detection_conf,
                "nms_threshold": nms_threshold,
            },
        }

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{self.api_url}/", json=payload, headers=headers
            )
            response.raise_for_status()

            data = response.json()
            if "error" in data:
                raise RuntimeError(f"Inference API Error: {data['error']}")

            return data.get("batch_faces", [])
