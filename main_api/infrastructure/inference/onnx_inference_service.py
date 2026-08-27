import httpx

from application.ports.inference import IInferenceService


class OnnxInferenceService(IInferenceService[str]):
    def __init__(self, api_url: str):
        self.api_url = api_url

    async def get_face_encodings(
        self,
        images: list[str],
        detection_conf: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> list[list[dict]]:
        headers = {"Content-Type": "application/json"}
        payload = {
            "inputs": images,
            "parameters": {
                "detection_conf": detection_conf,
                "nms_threshold": nms_threshold,
            },
        }

        import logging
        import time

        logger = logging.getLogger(__name__)
        # using a 60s timeout to prevent hanging if inference server dies
        async with httpx.AsyncClient(timeout=150.0) as client:
            t0 = time.time()
            response = await client.post(f"{self.api_url}/", json=payload, headers=headers)
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
