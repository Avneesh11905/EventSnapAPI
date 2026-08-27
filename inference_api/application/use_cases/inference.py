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
        self, inputs: list[str] | list[bytes], parameters: InferenceParametersDTO
    ) -> InferenceResultDTO:
        max_faces = parameters.max_faces
        detection_conf = parameters.detection_conf
        nms_thresh = parameters.nms_threshold

        try:
            import time

            t0 = time.perf_counter()
            cv_images = self.image_decoder.decode_batch(inputs)
            t_decode = time.perf_counter()
            logger.info(f"decode_batch took: {t_decode - t0:.4f}s")

            batch_faces = self.detector.detect_batch(
                cv_images,
                max_faces=max_faces,
                confidence=detection_conf,
                nms_threshold=nms_thresh,
            )
            t_detect = time.perf_counter()
            logger.info(f"detect_batch took: {t_detect - t_decode:.4f}s")

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

            t_align = time.perf_counter()
            logger.info(f"align took: {t_align - t_detect:.4f}s")

            if all_aligned_faces:
                all_embeddings = self.embedder.embed_batch(all_aligned_faces)
                t_embed = time.perf_counter()
                logger.info(f"embed_batch took: {t_embed - t_align:.4f}s")

                for (img_idx, face_idx), emb in zip(face_mapping, all_embeddings):
                    face_obj = batch_faces[img_idx][face_idx]
                    final_results[img_idx].append(
                        FaceResultDTO(
                            bbox=face_obj.bbox.tolist(),
                            confidence=float(face_obj.confidence),
                            embedding=emb.tolist(),
                        )
                    )

            t_final = time.perf_counter()
            logger.info(
                f"map_results took: {t_final - (t_embed if all_aligned_faces else t_align):.4f}s"
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
