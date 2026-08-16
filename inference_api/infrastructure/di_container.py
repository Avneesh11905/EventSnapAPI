from dependency_injector import containers, providers
from application.use_cases.inference import ProcessImagesUseCase
from infrastructure.onnx.detector import FaceDetector
from infrastructure.onnx.embedder import FaceEmbedder
from infrastructure.image_decoder import Base64ImageDecoder


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "presentation.api.routes.inference",
        ]
    )

    face_detector = providers.Singleton(
        FaceDetector,
        model_path="models/buffalo_l/det_10g.onnx",
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


_container_instance = None


def get_container() -> Container:
    global _container_instance
    if _container_instance is None:
        _container_instance = Container()
        _container_instance.wire()
    return _container_instance
