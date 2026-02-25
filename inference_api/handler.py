import base64
import json
import logging
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from detector import FaceDetector
from embedder import FaceEmbedder

logger = logging.getLogger(__name__)


class EndpointHandler:
    """
    Custom inference handler for Hugging Face Inference Endpoints.
    Provides a stateless API for face detection and embedding.
    """

    def __init__(self, path: str = "."):
        """
        Initializes the model endpoints. `path` is the repository root path
        in the Hugging Face container.
        """
        logger.info("Initializing Eventsnap EndpointHandler...")

        # Initialize detector
        det_path = f"{path}/models/det_10g.onnx"
        self.detector = FaceDetector(
            model_path=det_path,
            device="cuda",  # HF Endpoints uses CUDA if selected
        )

        # Initialize embedder
        emb_path = f"{path}/models/glintr100.onnx"
        self.embedder = FaceEmbedder(
            model_path=emb_path,
            device="cuda",
        )
        logger.info("Models loaded successfully.")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Endpoint core logic. Receives JSON dict matching the API request format
        sent by main_api orchestrator.
        """
        inputs = data.get("inputs", data)
        parameters = data.get("parameters", {})

        max_faces_param = parameters.get("max_faces", 0)
        max_faces = 0 if max_faces_param == "all" else int(max_faces_param)
        detection_conf = float(parameters.get("detection_conf", 0.5))
        nms_thresh = float(parameters.get("nms_threshold", 0.4))

        original_conf = self.detector.confidence
        original_nms = self.detector.nms_threshold
        
        self.detector.confidence = detection_conf
        self.detector.nms_threshold = nms_thresh

        try:
            is_batch = isinstance(inputs, list)
            input_strings = inputs if is_batch else [inputs]
            
            cv_images = []
            for b64_str in input_strings:
                if not isinstance(b64_str, str):
                    return {"error": "Invalid input format."}
                
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_str)
                pil_image = Image.open(BytesIO(image_bytes))
                
                # Check for Grayscale to avoid shape issues
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                    
                cv_image = np.array(pil_image)
                if cv_image.shape[-1] == 4:
                    cv_image = cv_image[..., :3]
                cv_image = cv_image[:, :, ::-1].copy()
                cv_images.append(cv_image)

            # 2. Detect Faces
            batch_faces = self.detector.detect_batch(cv_images, max_faces=max_faces)

            # 3. Extract Embeddings (Flattened Batching)
            final_results = [[] for _ in range(len(cv_images))]
            all_aligned_faces = []
            face_mapping = [] # (img_idx, face_idx_within_img)
            
            for img_idx, (cv_image, faces) in enumerate(zip(cv_images, batch_faces)):
                if faces:
                    for face_idx, face in enumerate(faces):
                        aligned = self.embedder.align(cv_image, face.landmarks)
                        all_aligned_faces.append(aligned)
                        face_mapping.append((img_idx, face_idx))

            if all_aligned_faces:
                # Embed ALL faces from ALL images in one giant batch call to saturate GPU
                all_embeddings = self.embedder.embed_batch(all_aligned_faces)
                
                # Distribute embeddings back to their respective origin images
                for (img_idx, face_idx), emb in zip(face_mapping, all_embeddings):
                    face_obj = batch_faces[img_idx][face_idx]
                    final_results[img_idx].append({
                        "bbox": face_obj.bbox.tolist(),
                        "confidence": float(face_obj.confidence),
                        "embedding": emb.tolist()
                    })

            if not is_batch:
                return {"faces": final_results[0]}
            else:
                return {"batch_faces": final_results}

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e)}
        finally:
            self.detector.confidence = original_conf
            self.detector.nms_threshold = original_nms
