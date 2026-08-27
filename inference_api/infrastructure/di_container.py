from dependency_injector import containers, providers
from application.use_cases.inference import ProcessImagesUseCase
from infrastructure.onnx.detector import FaceDetector
from infrastructure.onnx.embedder import FaceEmbedder
from infrastructure.image_decoder import Base64ImageDecoder, BytesImageDecoder


from concurrent.futures import ThreadPoolExecutor


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "presentation.api.routes.inference",
        ]
    )

    # Shared thread pool to ensure ONNX models (not thread-safe)
    # only process one request at a time across both HTTP and gRPC.
    inference_executor = providers.Singleton(
        ThreadPoolExecutor,
        max_workers=1,
    )

    face_detector = providers.Singleton(
        FaceDetector,
        model_path="models/buffalo_l/det_10g.onnx",
        input_size=(1024, 1024),
        device="cuda",
        num_threads=1,
    )

    face_embedder = providers.Singleton(
        FaceEmbedder,
        model_path="models/antelopev2/glintr100.onnx",
        device="cuda",
        num_threads=1,
    )

    image_decoder = providers.Singleton(Base64ImageDecoder)

    process_images_use_case = providers.Factory(
        ProcessImagesUseCase,
        detector=face_detector,
        embedder=face_embedder,
        image_decoder=image_decoder,
    )

    bytes_image_decoder = providers.Singleton(BytesImageDecoder)

    process_images_bytes_use_case = providers.Factory(
        ProcessImagesUseCase,
        detector=face_detector,
        embedder=face_embedder,
        image_decoder=bytes_image_decoder,
    )


_container_instance = None


def get_container() -> Container:
    global _container_instance
    if _container_instance is None:
        _container_instance = Container()
        _container_instance.wire()
    return _container_instance
