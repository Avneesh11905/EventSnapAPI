import pytest
import cv2
import base64
import httpx
import grpc
from grpc import aio

from presentation.grpc.proto import inference_pb2, inference_pb2_grpc


@pytest.fixture(scope="module")
def real_1024_image_bytes():
    """Reads a real image from the disk and resizes it exactly to 1024x1024."""
    image_path = r"d:\EventSnap\EventSnapAPI\images\PXL_20241116_133746627.jpg"
    img = cv2.imread(image_path)
    if img is None:
        pytest.skip(f"Could not load real image at {image_path}")
    
    img_1024 = cv2.resize(img, (1024, 1024))
    _, buffer = cv2.imencode('.jpg', img_1024)
    return buffer.tobytes()


@pytest.fixture(scope="module")
def real_1024_image_base64(real_1024_image_bytes: bytes) -> str:
    return base64.b64encode(real_1024_image_bytes).decode("utf-8")


@pytest.mark.asyncio
async def test_inference_grpc_real_batch_64(real_1024_image_bytes: bytes):
    """
    Integration test: Hits the live gRPC API in Docker with a batch of 64 images.
    """
    channel_options = [
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ("grpc.max_send_message_length", 256 * 1024 * 1024),
    ]
    
    async with aio.insecure_channel("localhost:50051", options=channel_options) as channel:
        stub = inference_pb2_grpc.FaceInferenceStub(channel)
        batch_size = 64
        
        request = inference_pb2.InferenceRequest(
            images=[real_1024_image_bytes] * batch_size,
            detection_conf=0.5,
            nms_threshold=0.4,
            max_faces=0
        )
        
        response = await stub.ExtractFaces(request, timeout=60.0)
        
        assert len(response.results) == batch_size
        assert len(response.results[0].faces) > 0


@pytest.mark.asyncio
async def test_inference_http_real_batch_64(real_1024_image_base64: str):
    """
    Integration test: Hits the live HTTP API in Docker with a batch of 64 images.
    """
    batch_size = 64
    payload = {
        "inputs": [real_1024_image_base64] * batch_size,
        "parameters": {
            "max_faces": "0",
            "detection_conf": 0.5,
            "nms_threshold": 0.4
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/", json=payload, timeout=60.0)
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_faces" in data
        assert len(data["batch_faces"]) == batch_size
        assert len(data["batch_faces"][0]) > 0
