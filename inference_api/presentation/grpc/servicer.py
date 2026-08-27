import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import grpc
from grpc import aio

from application.dtos import InferenceParametersDTO
from infrastructure.di_container import Container

from presentation.grpc.proto import inference_pb2
from presentation.grpc.proto import inference_pb2_grpc

logger = logging.getLogger(__name__)



class FaceInferenceServicer(inference_pb2_grpc.FaceInferenceServicer):
    """
    gRPC servicer for face inference.

    Receives raw image bytes (no base64), runs the same ProcessImagesUseCase
    used by the HTTP route (but wired with BytesImageDecoder), and returns
    structured face results.
    """

    def __init__(self, container: Container) -> None:
        self._container = container

    async def ExtractFaces(
        self,
        request: inference_pb2.InferenceRequest,
        context: aio.ServicerContext,
    ) -> inference_pb2.InferenceResponse:
        images_bytes: list[bytes] = list(request.images)

        if not images_bytes:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "Request must contain at least one image."
            )

        detection_conf = request.detection_conf if request.detection_conf else 0.5
        nms_threshold = request.nms_threshold if request.nms_threshold else 0.4
        max_faces = request.max_faces  # 0 means unlimited

        params = InferenceParametersDTO(
            max_faces=max_faces,
            detection_conf=detection_conf,
            nms_threshold=nms_threshold,
        )

        # Resolve a fresh use-case instance (Factory provider) from the container.
        # The underlying Singleton detector/embedder are shared with the HTTP path.
        use_case = self._container.process_images_bytes_use_case()

        loop = asyncio.get_running_loop()
        
        result = await loop.run_in_executor(
            self._container.inference_executor(),
            use_case.execute,
            images_bytes,
            params,
        )

        # Map domain DTOs → protobuf messages
        response = inference_pb2.InferenceResponse()
        for image_faces in result.batch_faces:
            faces_proto = [
                inference_pb2.FaceResult(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    embedding=face.embedding,
                ) for face in image_faces
            ]
            response.results.append(inference_pb2.ImageFaces(faces=faces_proto))

        return response
