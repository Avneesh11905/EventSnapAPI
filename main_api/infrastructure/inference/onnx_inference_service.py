from application.ports.inference import IInferenceService
import httpx
from typing import List, Dict


class OnnxInferenceService(IInferenceService):
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

        import time
        import logging
        logger = logging.getLogger(__name__)

        async with httpx.AsyncClient(timeout=None) as client:
            t0 = time.time()
            response = await client.post(
                f"{self.api_url}/", json=payload, headers=headers
            )
            t1 = time.time()
            logger.info(f"httpx.post (headers+body) took {t1 - t0:.2f}s")
            
            response.raise_for_status()

            t2 = time.time()
            data = response.json()
            t3 = time.time()
            logger.info(f"response.json() parsing took {t3 - t2:.2f}s")

            if "error" in data:
                raise RuntimeError(f"Inference API Error: {data['error']}")

            return data.get("batch_faces", [])
