"""
Face Detection Module — SCRFD (ONNX)

Uses InsightFace's SCRFD (det_500m) for fast, lightweight face detection via ONNX Runtime.
Post-processing based on the official InsightFace SCRFD implementation.
No torch/ultralytics dependency needed.

Returns bounding boxes and 5-point facial landmarks for each detected face.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class DetectedFace:
    """A single detected face with its bounding box and landmarks."""
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixel coords
    landmarks: np.ndarray     # shape (5, 2) — left_eye, right_eye, nose, left_mouth, right_mouth
    confidence: float

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

class FaceDetector:
    """
    SCRFD-based face detector using ONNX Runtime.

    Args:
        model_path: Path to det_500m.onnx model.
        confidence: Minimum detection confidence (0-1).
        nms_threshold: NMS IoU threshold for overlapping detections.
        input_size: Model input size (width, height). Default (640, 640).
    """

    _FEAT_STRIDE_FPN = [8, 16, 32]
    _NUM_ANCHORS = 2
    _FMC = 3  # number of feature map levels

    def __init__(
        self,
        model_path: str = "models/buffalo_l/det_10g.onnx",
        confidence: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
        device: str = "cpu",
        num_threads: int = 0,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Detection model not found at '{model_path}'. "
                f"Run download_models.py first."
            )

        providers = _get_providers(device)
        sess_opts = ort.SessionOptions()
        if num_threads > 0:
            sess_opts.intra_op_num_threads = num_threads
            sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.input_size = input_size  # (width, height)

        # Cache anchor centers per (height, width, stride)
        self._center_cache: dict[tuple, np.ndarray] = {}
        
        # Check if model supports batching
        input_shape = self.session.get_inputs()[0].shape
        # Typically [batch, channel, height, width]
        # If dim 0 is a string (e.g. 'batch') or None, it's dynamic.
        # If dim 0 is an int (e.g. 1), it's fixed.
        self.batch_supported = False
        if len(input_shape) > 0:
            dim0 = input_shape[0]
            if isinstance(dim0, str) or dim0 is None:
                self.batch_supported = True
            elif isinstance(dim0, int) and dim0 > 1:
                 # Fixed batch size > 1? Rare but possible.
                 pass
        
        if not self.batch_supported:
            print(f"[FaceDetector] Model has fixed batch size {input_shape[0]}. Batch inference will be sequential.")


    @staticmethod
    def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """Decode distance predictions to bounding boxes."""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """Decode distance predictions to keypoints."""
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def _nms(self, dets: np.ndarray) -> list[int]:
        """Non-maximum suppression on (N, 5) array of [x1, y1, x2, y2, score]."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Resize and normalize image for SCRFD.

        Returns:
            (input_blob, det_scale)
        """
        h, w = image.shape[:2]
        input_w, input_h = self.input_size

        # Calculate scale
        im_ratio = float(h) / w
        model_ratio = float(input_h) / input_w
        if im_ratio > model_ratio:
            new_height = input_h
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_w
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / h

        resized = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized

        # Use cv2.dnn.blobFromImage for correct preprocessing (matches official impl)
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, (input_w, input_h),
            (127.5, 127.5, 127.5), swapRB=True
        )

        return blob, det_scale

    def _get_anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        """Get or compute cached anchor centers for a feature map."""
        key = (height, width, stride)
        if key in self._center_cache:
            return self._center_cache[key]

        anchor_centers = np.stack(
            np.mgrid[:height, :width][::-1], axis=-1
        ).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape(-1, 2)

        if self._NUM_ANCHORS > 1:
            anchor_centers = np.stack(
                [anchor_centers] * self._NUM_ANCHORS, axis=1
            ).reshape(-1, 2)

        if len(self._center_cache) < 100:
            self._center_cache[key] = anchor_centers

        return anchor_centers

    def detect(self, image: np.ndarray, max_faces: int = 0) -> list[DetectedFace]:
        """
        Detect faces in an image.

        Args:
            image: BGR numpy array (OpenCV format).
            max_faces: Maximum number of faces to return. 0 = unlimited.

        Returns:
            List of DetectedFace sorted by confidence (highest first).
        """
        blob, det_scale = self._preprocess(image)

        # Run inference
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        scores_list = []
        bboxes_list = []
        kpss_list = []

        fmc = self._FMC

        for idx, stride in enumerate(self._FEAT_STRIDE_FPN):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride  # scale by stride
            kps_preds = net_outs[idx + fmc * 2] * stride  # scale by stride

            height = input_height // stride
            width = input_width // stride

            anchor_centers = self._get_anchor_centers(height, width, stride)

            # Filter by confidence
            pos_inds = np.where(scores >= self.confidence)[0]

            # Decode all, then filter
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            kpss = self._distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape(kpss.shape[0], -1, 2)

            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            pos_kpss = kpss[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss_list.append(pos_kpss)

        if not scores_list or all(len(s) == 0 for s in scores_list):
            return []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale

        # NMS
        pre_det = np.hstack((bboxes, scores)).astype(np.float32)
        pre_det = pre_det[order, :]
        keep = self._nms(pre_det)
        det = pre_det[keep, :]

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        # Limit max faces
        if max_faces > 0 and det.shape[0] > max_faces:
            det = det[:max_faces]
            kpss = kpss[:max_faces]

        # Build results
        faces: list[DetectedFace] = []
        for i in range(det.shape[0]):
            face = DetectedFace(
                bbox=det[i, :4].astype(np.int32),
                landmarks=kpss[i].astype(np.float32),
                confidence=float(det[i, 4]),
            )
            faces.append(face)

        return faces

    def detect_batch(self, images: list[np.ndarray], max_faces: int = 0) -> list[list[DetectedFace]]:
        """
        Detect faces in a batch of images.
        """
        if not images:
            return []

        if not self.batch_supported:
            import concurrent.futures
            
            # Determine an optimal pool size for concurrent pseudo-batching
            # Limited to 4 workers to prevent CUBLAS_STATUS_ALLOC_FAILED (GPU OOM) on CUDA
            workers = min(len(images), 4)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all images to be detected concurrently
                futures = [executor.submit(self.detect, img, max_faces) for img in images]
                
                # Gather results in the exact order they were submitted
                results = [future.result() for future in futures]
                
            return results

        batch_size = len(images)
        input_w, input_h = self.input_size

        # Prepare batch blob
        batch_blob = np.zeros((batch_size, 3, input_h, input_w), dtype=np.float32)
        det_scales = []

        for i, img in enumerate(images):
            h, w = img.shape[:2]
            im_ratio = float(h) / w
            model_ratio = float(input_h) / input_w

            if im_ratio > model_ratio:
                new_height = input_h
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_w
                new_height = int(new_width * im_ratio)

            det_scale = float(new_height) / h
            det_scales.append(det_scale)

            resized = cv2.resize(img, (new_width, new_height))

            # 1. Embed in canvas
            canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
            # Center? Or top-left? InsightFace usually top-left.
            canvas[:new_height, :new_width, :] = resized

            # 2. BGR -> RGB & Normalize
            canvas = canvas.astype(np.float32)
            canvas = (canvas - 127.5) / 128.0

            # 3. BGR -> RGB
            canvas = canvas[..., ::-1]

            # 4. HWC -> CHW
            canvas = np.transpose(canvas, (2, 0, 1))

            batch_blob[i] = canvas

        # Run inference
        net_outs = self.session.run(self.output_names, {self.input_name: batch_blob})

        # Process batch results
        batch_results = []
        fmc = self._FMC

        for b in range(batch_size):
            scores_list = []
            bboxes_list = []
            kpss_list = []
            
            for idx, stride in enumerate(self._FEAT_STRIDE_FPN):
                # net_outs[idx] is (B, A*C, H, W) or (B, C, H, W)?
                # SCRFD dynamic batch output usually conserves dimensions.
                # (B, NumAnchors*1, H, W) for scores
                
                # We need to slice the batch
                score_blob = net_outs[idx][b]      # (C, H, W)
                bbox_blob = net_outs[idx + fmc][b] # (C*4, H, W)
                kps_blob = net_outs[idx + fmc * 2][b] # (C*10, H, W)
                
                c, h_map, w_map = score_blob.shape
                
                # Transpose to (H, W, C) & Reshape
                # Note: SCRFD C=NumAnchors * 1
                
                # Flatten
                score_blob = score_blob.transpose(1, 2, 0).reshape(-1, 1)
                bbox_blob = bbox_blob.transpose(1, 2, 0).reshape(-1, 4)
                kps_blob = kps_blob.transpose(1, 2, 0).reshape(-1, 10)
                
                anchor_centers = self._get_anchor_centers(h_map, w_map, stride)
                
                # Filter
                pos_inds = np.where(score_blob >= self.confidence)[0]
                
                # Decode
                bbox_pred = bbox_blob[pos_inds] * stride
                kps_pred = kps_blob[pos_inds] * stride
                
                anchors = anchor_centers[pos_inds]
                
                if len(anchors) > 0:
                     bboxes = self._distance2bbox(anchors, bbox_pred)
                     kpss = self._distance2kps(anchors, kps_pred)
                     
                     scores_list.append(score_blob[pos_inds])
                     bboxes_list.append(bboxes)
                     kpss_list.append(kpss.reshape(-1, 5, 2))
                
            if not scores_list:
                batch_results.append([])
                continue
                
            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]

            bboxes = np.vstack(bboxes_list) / det_scales[b]
            kpss = np.vstack(kpss_list) / det_scales[b]
            
            # NMS
            pre_det = np.hstack((bboxes, scores)).astype(np.float32)
            pre_det = pre_det[order, :]
            keep = self._nms(pre_det)
            det = pre_det[keep, :]
            
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
            
            if max_faces > 0 and det.shape[0] > max_faces:
                det = det[:max_faces]
                kpss = kpss[:max_faces]
                
            faces = []
            for i in range(det.shape[0]):
                face = DetectedFace(
                    bbox=det[i, :4].astype(np.int32),
                    landmarks=kpss[i].astype(np.float32),
                    confidence=float(det[i, 4]),
                )
                faces.append(face)
            batch_results.append(faces)
            
        return batch_results