import pytest
import grpc
from grpc import aio

from presentation.grpc.servicer import FaceInferenceServicer
from presentation.grpc.interceptors import ExceptionInterceptor
from presentation.grpc.proto import inference_pb2, inference_pb2_grpc


@pytest.fixture
async def grpc_test_server(mock_container):
    """
    Spins up an ephemeral, in-memory gRPC server bound to a random free port.
    This lets us test the gRPC handler completely isolated from the main app.
    """
    server = aio.server(interceptors=[ExceptionInterceptor()])
    servicer = FaceInferenceServicer(mock_container)
    inference_pb2_grpc.add_FaceInferenceServicer_to_server(servicer, server)

    port = server.add_insecure_port("localhost:0")
    await server.start()

    yield f"localhost:{port}"

    await server.stop(grace=0)


@pytest.mark.asyncio
async def test_inference_grpc_success(grpc_test_server, dummy_image_bytes):
    """Tests the happy path of the gRPC ExtractFaces RPC."""
    async with aio.insecure_channel(grpc_test_server) as channel:
        stub = inference_pb2_grpc.FaceInferenceStub(channel)

        request = inference_pb2.InferenceRequest(
            images=[dummy_image_bytes],
            detection_conf=0.5,
            nms_threshold=0.4,
            max_faces=0,
        )

        response = await stub.ExtractFaces(request)

        # 1 image sent -> 1 ImageFaces result
        assert len(response.results) == 1

        image_faces = response.results[0]
        # 1 face mocked per image
        assert len(image_faces.faces) == 1

        face = image_faces.faces[0]
        assert len(face.bbox) == 4
        assert face.confidence > 0.0
        assert len(face.embedding) == 512


@pytest.mark.asyncio
async def test_inference_grpc_invalid_image(grpc_test_server):
    """Tests that sending garbage bytes triggers a gRPC INVALID_ARGUMENT abort."""
    async with aio.insecure_channel(grpc_test_server) as channel:
        stub = inference_pb2_grpc.FaceInferenceStub(channel)

        request = inference_pb2.InferenceRequest(
            images=[b"not-a-valid-image-byte-stream"]
        )

        with pytest.raises(grpc.RpcError) as exc_info:
            await stub.ExtractFaces(request)

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
