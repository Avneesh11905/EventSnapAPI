import logging
from application.ports.face_services import IFaceDetector, IFaceEmbedder
from application.ports.image_services import IImageDecoder
from application.dtos import InferenceParametersDTO, InferenceResultDTO, FaceResultDTO

logger = logging.getLogger(__name__)


class ProcessImagesUseCase:
    def __init__(
        self,
        detector: IFaceDetector,
        embedder: IFaceEmbedder,
        image_decoder: IImageDecoder,
    ):
        self.detector = detector
        self.embedder = embedder
        self.image_decoder = image_decoder

    def execute(
        self, inputs: list[str], parameters: InferenceParametersDTO
    ) -> InferenceResultDTO:
        max_faces = parameters.max_faces
        detection_conf = parameters.detection_conf
        nms_thresh = parameters.nms_threshold

        try:
            logger.info(f"Processing inference for {len(inputs)} images")

            cv_images = self.image_decoder.decode_batch(inputs)

            batch_faces = self.detector.detect_batch(
                cv_images,
                max_faces=max_faces,
                confidence=detection_conf,
                nms_threshold=nms_thresh,
            )

            final_results: list[list[FaceResultDTO]] = [
                [] for _ in range(len(cv_images))
            ]
            all_aligned_faces = []
            face_mapping = []

            for img_idx, (cv_image, faces) in enumerate(zip(cv_images, batch_faces)):
                if faces:
                    for face_idx, face in enumerate(faces):
                        aligned = self.embedder.align(cv_image, face.landmarks)
                        all_aligned_faces.append(aligned)
                        face_mapping.append((img_idx, face_idx))

            if all_aligned_faces:
                all_embeddings = self.embedder.embed_batch(all_aligned_faces)

                for (img_idx, face_idx), emb in zip(face_mapping, all_embeddings):
                    face_obj = batch_faces[img_idx][face_idx]
                    final_results[img_idx].append(
                        FaceResultDTO(
                            bbox=face_obj.bbox.tolist(),
                            confidence=float(face_obj.confidence),
                            embedding=emb.tolist(),
                        )
                    )

            return InferenceResultDTO(batch_faces=final_results)

        finally:
            if "cv_images" in locals():
                del cv_images
            if "batch_faces" in locals():
                del batch_faces
            if "all_aligned_faces" in locals():
                del all_aligned_faces
            if "all_embeddings" in locals():
                del all_embeddings

            import gc

            gc.collect()
