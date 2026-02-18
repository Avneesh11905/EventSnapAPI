"""
Face Embedding Module — MobileFaceNet (ONNX)

Handles face alignment (affine transform using 5-point landmarks)
and embedding extraction via ONNX Runtime.
Produces a normalized 512-d vector that encodes facial identity.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# Standard reference landmarks for ArcFace-style alignment (112x112)
# These are the "ideal" positions for: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def _get_providers(device: str) -> list:
    """Get ONNX Runtime execution providers based on device."""
    if device == "cuda":
        available = ort.get_available_providers()
        providers = []
        # if "TensorrtExecutionProvider" in available:
        #     providers.append(("TensorrtExecutionProvider", {
        #         "trt_fp16_enable": True,
        #         "trt_engine_cache_enable": True,
        #         "trt_engine_cache_path": "./trt_cache",
        #     }))
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            }))
        if providers:
            providers.append("CPUExecutionProvider")
            return providers
        print("[warning] CUDA requested but no GPU providers available, falling back to CPU")
    return ["CPUExecutionProvider"]


class FaceEmbedder:
    """
    ONNX-based face embedding extractor.

    Args:
        model_path: Path to MobileFaceNet ONNX model.
        input_size: Model input size (width, height). Default (112, 112).
    """

    def __init__(
        self,
        model_path: str = "models/antelopev2/glintr100.onnx",
        input_size: tuple[int, int] = (112, 112),
        device: str = "cpu",
        num_threads: int = 0,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Embedding model not found at '{model_path}'. "
                f"Run download_models.py first."
            )

        providers = _get_providers(device)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3

        if num_threads > 0:
            sess_opts.intra_op_num_threads = num_threads
            sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align a face using 5-point landmarks via affine transform.

        Args:
            image: Full BGR image (numpy array).
            landmarks: Shape (5, 2) — detected facial landmarks.

        Returns:
            Aligned face crop, shape (112, 112, 3), BGR, uint8.
        """
        assert landmarks.shape == (5, 2), f"Expected (5,2) landmarks, got {landmarks.shape}"

        # Estimate affine transform from detected landmarks to reference
        # Using 3 points (both eyes + nose) for a stable affine transform
        src_pts = landmarks[:3].astype(np.float32)
        dst_pts = ARCFACE_REF_LANDMARKS[:3].astype(np.float32)

        M = cv2.getAffineTransform(src_pts, dst_pts)
        aligned = cv2.warpAffine(
            image, M,
            (self.input_size[0], self.input_size[1]),
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned

    def preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Preprocess an aligned face for the ONNX model.

        Args:
            aligned_face: BGR image, shape (112, 112, 3), uint8.

        Returns:
            Float32 tensor, shape (1, 3, 112, 112), normalized to [-1, 1].
        """
        # BGR → RGB
        img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1]
        img = (img.astype(np.float32) - 127.5) / 127.5
        # HWC → CHW → NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for a batch of aligned faces.

        Args:
            aligned_faces: List of BGR images, each (112, 112, 3), uint8.

        Returns:
            Array of L2-normalized embeddings, shape (N, 512).
        """
        if not aligned_faces:
            return np.empty((0, 512), dtype=np.float32)

        # Process in chunks to avoid OOM
        chunk_size = 32
        embeddings_list = []
        
        for i in range(0, len(aligned_faces), chunk_size):
            chunk = aligned_faces[i : i + chunk_size]
            
            # 1. Stack: (N, 112, 112, 3)
            blob = np.stack(chunk)
            
            # 2. BGR -> RGB
            blob = blob[..., ::-1]  # efficient BGR->RGB
            
            # 3. Normalize
            blob = (blob.astype(np.float32) - 127.5) / 127.5
            
            # 4. NHWC -> NCHW
            blob = np.transpose(blob, (0, 3, 1, 2))
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: blob})
            emb_chunk = outputs[0]
            
            # L2 normalize
            norms = np.linalg.norm(emb_chunk, axis=1, keepdims=True)
            emb_chunk = emb_chunk / (norms + 1e-10)
            
            embeddings_list.append(emb_chunk)
            
        return np.vstack(embeddings_list)
