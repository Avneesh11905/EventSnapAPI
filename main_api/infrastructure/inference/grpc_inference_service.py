import logging

import grpc
from grpc import aio

from application.ports.inference import IInferenceService
from infrastructure.inference.proto import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)


class GrpcInferenceService(IInferenceService[bytes]):
    def __init__(self, api_url: str):
        # api_url should be like "es-inference-api:50051"
        self.api_url = api_url
        self.channel_options = [
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ]

    async def get_face_encodings(
        self,
        images: list[bytes],
        detection_conf: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> list[list[dict]]:

        is_secure = self.api_url.endswith(":443")
        
        channel_mgr = (
            aio.secure_channel(self.api_url, grpc.ssl_channel_credentials(), options=self.channel_options)
            if is_secure
            else aio.insecure_channel(self.api_url, options=self.channel_options)
        )

        async with channel_mgr as channel:
            stub = inference_pb2_grpc.FaceInferenceStub(channel)

            request = inference_pb2.InferenceRequest(  # type: ignore
                images=images,
                detection_conf=detection_conf,
                nms_threshold=nms_threshold,
                max_faces=0,  # 0 means "all faces"
            )

            try:
                response = await stub.ExtractFaces(request, timeout=300.0)
            except grpc.aio.AioRpcError as e:
                logger.error(f"[gRPC] Inference failed: {e.details()}")
                raise RuntimeError(f"Inference API Error: {e.details()}") from e

            # Map protobuf response back to domain list[list[dict]]
            results: list[list[dict]] = []
            for image_result in response.results:  # type: ignore
                faces = []
                for face in image_result.faces:
                    faces.append(
                        {
                            "bbox": list(face.bbox),
                            "confidence": face.confidence,
                            "embedding": list(face.embedding),
                        }
                    )
                results.append(faces)

            return results

