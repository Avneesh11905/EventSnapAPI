from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.inference.onnx_inference_service import OnnxInferenceService


@pytest.fixture
def onnx_service():
    return OnnxInferenceService("http://localhost:8001")


@pytest.mark.asyncio
async def test_get_face_encodings(onnx_service):
    with patch(
        "infrastructure.inference.onnx_inference_service.httpx.AsyncClient"
    ) as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "batch_faces": [[{"embedding": [0.1, 0.2], "confidence": 0.99}]]
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response

        res = await onnx_service.get_face_encodings([b"img1"])
        assert len(res) == 1
        assert res[0][0]["embedding"] == [0.1, 0.2]
