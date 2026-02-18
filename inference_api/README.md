# Eventsnap Inference API (GPU Worker)

The `inference_api` is an ultra-fast, stateless FastAPI server designed purely for mathematical facial processing. It uses the `insightface` library powered by `onnxruntime-gpu` (CUDA 11) to extract 512-dimension vector embeddings from raw image data.

It is designed to be horizontally scaled or deployed as a Serverless Endpoint on Hugging Face (e.g., Nvidia T4 hardware).

## Core Responsibilities
*   **Vectorization**: Converts human faces into mathematical matrices (ResNet100 model).
*   **Stateless Processing**: Receives base64 strings, returns facial coordinates and embeddings, and holds nothing in memory.

---

## REST Endpoints

### 1. Process Batch Images
Accepts an array of Base64 encoded images, passes them through the SCRFD detection model and arcface recognition model, and returns comprehensive facial data (bounding boxes, keypoints, confidence, and embeddings).

**Example `curl` Request:**
```bash
curl -X 'POST' \
  'http://localhost:5000/' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    "base64_encoded_photo_1...",
    "base64_encoded_photo_2..."
  ],
  "max_faces": 0,
  "det_conf": 0.5,
  "nms_thresh": 0.4
}'
```

**Example `axios` (Next.js) Request:**
```javascript
import axios from 'axios';

const response = await axios.post('http://localhost:5000/', {
  inputs: [
    'base64_encoded_photo_1...',
    'base64_encoded_photo_2...'
  ],
  max_faces: 0,
  det_conf: 0.5,
  nms_thresh: 0.4
}, {
  headers: {
    'Content-Type': 'application/json'
  }
});
console.log(response.data);
```

**Response (200 OK):**
```json
[
  [ // Image 1 (2 faces found)
    {
      "bbox": [104.2, 210.5, 305.1, 401.3],
      "kps": [[150.1, 250.2], ...],
      "det_score": 0.998,
      "embedding": [0.012, -0.054, ...] // 512 float array
    },
    { ... }
  ],
  [ // Image 2 (0 faces found)
  ] 
]
```

## Environment & Build

Because ONNX Runtime GPU requires specific NVIDIA compilation, the container must be built utilizing `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` and executed on a host machine with NVIDIA Drivers and the `nvidia-container-toolkit` installed.

To run locally (if you have an NVIDIA GPU):
```bash
docker run --gpus all -p 5000:5000 avneesh11905/inference_api:latest
```
