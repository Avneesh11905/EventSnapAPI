from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.inference.grpc_inference_service import GrpcInferenceService


@pytest.fixture
def grpc_service():
    return GrpcInferenceService("localhost:50051")


@pytest.mark.asyncio
async def test_get_face_encodings(grpc_service):
    with patch("infrastructure.inference.grpc_inference_service.grpc.aio.insecure_channel"):
        mock_stub = AsyncMock()
        with patch(
            "infrastructure.inference.grpc_inference_service.inference_pb2_grpc.FaceInferenceStub",
            return_value=mock_stub,
        ):
            mock_response = MagicMock()
            mock_image_result = MagicMock()
            mock_face = MagicMock()
            mock_face.embedding = [0.1, 0.2]
            mock_face.confidence = 0.99
            mock_face.bbox = [0, 0, 10, 10]
            mock_image_result.faces = [mock_face]
            mock_response.results = [mock_image_result, mock_image_result]
            mock_stub.ExtractFaces.return_value = mock_response

            res = await grpc_service.get_face_encodings([b"img1", b"img2"])
            assert len(res) == 2
            assert res[0][0]["embedding"] == [0.1, 0.2]
            assert res[0][0]["confidence"] == 0.99
