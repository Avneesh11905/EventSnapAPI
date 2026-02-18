<div align="center">
  <h1>📸 Eventsnap API</h1>
  <i>A High-Performance Asynchronous Facial Recognition Pipeline</i><br>
  <i>Powered by FastAPI, Celery, PostgreSQL pgvector, and Hugging Face</i>
</div>

---

## 🚀 Overview

Eventsnap is a distributed, horizontally scalable microservice architecture designed to process hundreds of event photos simultaneously. It uses complex math to turn human faces into 512-dimension vectors, and stores them in a highly optimized vector database for instant facial matching.

It is split into three main components (each with their own dedicated README files):

### 1. `main_api` (The Orchestrator)
A FastAPI server that acts as the entry point. It accepts requests, authenticates them, dynamically creates Postgres tables, and dumps background encoding tasks into RabbitMQ for Celery workers to pick up.

### 2. `storage_api` (The Data Bridge)
A secondary asynchronous FastAPI server strictly meant for **local testing**. It provides an endpoint to upload and extract `.zip` test archives directly into your S3 (MinIO) buckets to simulate event uploads before moving to production.

### 3. `inference_api` (The GPU Worker)
A strictly mathematical, stateless ONNX Runtime container. It receives Base64 encoded photos, runs the powerful `insightface` SCRFD and ArcFace models on the NVIDIA GPU, and returns precise bounding boxes and 512-dimension `glintr100` embeddings.

---

## 🛠 Tech Stack

*   **API Framework:** FastAPI (Python 3.11, Native AsyncIO)
*   **Background Tasks:** Celery + RabbitMQ (Producer/Consumer model)
*   **Database:** PostgreSQL + [`pgvector`](https://github.com/pgvector/pgvector) extension (Cosine Similarity matching)
*   **Object Storage:** MinIO (S3 Compatible) via `aioboto3`
*   **Machine Learning:** ONNX runtime (CUDA 11.8), InsightFace
*   **Containerization:** Docker & Docker Compose

---

## ⚙️ How to Deploy Locally

Eventsnap is completely Dockerized for rapid development and testing.

### 1. Download the Heavy Models
Because the facial recognition models are too large for Git, you must download them before building the Docker containers.

Navigate sequentially into the `inference_api` folder and run the included Python script:
```bash
cd inference_api
python download_models.py
cd ..
```
*This will fetch the `det_10g.onnx` and `glintr100.onnx` binaries and place them in `inference_api/models/` for the Docker builder to copy.*

### 2. Build the Docker Images
Because the Python environment for facial recognition (CUDA, ONNX, OpenCV) is massive, we build the images independently to cache the layers effectively before orchestrating them.

```bash
# Build the stateless inference API (requires Nvidia Container Toolkit)
docker build -t avneesh11905/inference_api:latest ./inference_api

# Build the FastAPI orchestrator and Celery worker
docker build -t avneesh11905/main_api:latest ./main_api
```

### 3. Spin Up the Stack
Bring up all the containers (Postgres DB, RabbitMQ, MinIO, Inference API, Main API, and Celery Worker).

```bash
docker compose up -d
```

### 3. Monitor Your Cluster
*   **Main API Orchestrator:** http://localhost:8000/docs
*   **Storage API (Testing):** http://localhost:8001/docs
*   **Inference Model API:** http://localhost:5000/docs

---

## 📁 Architecture Flow

1.  A user uploads a ZIP of an event (`event_2026`) directly to the **Storage API** which extracts it into **MinIO**.
2.  The user hits the **Main API** `/encode-event/` endpoint.
3.  The **Main API** creates a Celery Task and immediately returns a `task_id` so the user isn't stuck waiting.
4.  The background **Celery Worker** picks up the task, pre-fetches images from **MinIO** using an aggressive 64-connection pool, beams them (Base64) to the **Inference API**, and bulk-inserts the generated 512D vectors directly into **PostgreSQL**.
5.  An attendee uploads 3 selfies to the **Main API** `/sort-attendee/` endpoint. The orchestrator gets the embeddings for those selfies, averages them, and executes a sub-millisecond `<=>` cosine similarity search in `pgvector` to find all photos they appear in!
