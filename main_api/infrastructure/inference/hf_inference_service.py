from application.ports.inference import InferenceService
import httpx
from typing import List, Dict

class HFInferenceService(InferenceService):
    def __init__(self, api_url: str, api_token: str | None = None):
        self.api_url = api_url
        self.api_token = api_token

    async def get_face_encodings(
        self, 
        b64_images: List[str], 
        max_faces: int = 0, 
        detection_conf: float = 0.5, 
        nms_threshold: float = 0.4
    ) -> List[List[Dict]]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        payload = {
            "inputs": b64_images,
            "parameters": {
                "max_faces": max_faces, 
                "detection_conf": detection_conf,
                "nms_threshold": nms_threshold
            }
        }
        
        import json
        import gzip
        
        payload_bytes = json.dumps(payload).encode('utf-8')
        compressed_payload = gzip.compress(payload_bytes)
        
        headers["Content-Encoding"] = "gzip"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_url}/", 
                content=compressed_payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            if "error" in data:
                 raise RuntimeError(f"Inference API Error: {data['error']}")
                 
            return data.get("batch_faces", [])
