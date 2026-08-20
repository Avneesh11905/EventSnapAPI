<div align="center">
  <h1>📸 Eventsnap API</h1>
  <i>A High-Performance Asynchronous Facial Recognition Pipeline</i><br>
  <i>Powered by FastAPI, Celery, PostgreSQL pgvector, and ONNX Runtime</i>
</div>

---

## 🚀 Overview

Eventsnap is a distributed, horizontally scalable microservice architecture designed to process hundreds of event photos simultaneously. It uses complex math to turn human faces into 512-dimension vectors, and stores them in a highly optimized vector database for instant facial matching.

It is split into two main components (each with their own dedicated README files):

### 1. `main_api` (The Orchestrator)
A FastAPI server that acts as the entry point. It accepts requests, authenticates them, saves face embeddings to Postgres, and dumps background encoding tasks into RabbitMQ for Celery workers to pick up.

### 2. `inference_api` (The GPU Worker)
A strictly mathematical, stateless ONNX Runtime container. It receives Base64 encoded photos, runs the powerful `insightface` SCRFD and ArcFace models on the NVIDIA GPU, and returns precise bounding boxes and 512-dimension `glintr100` embeddings.

---

## 🛠 Tech Stack

*   **API Framework:** FastAPI (Python 3.14, Native AsyncIO)
*   **Architecture Pattern:** Hexagonal Architecture (Ports and Adapters) with Dependency Injection
*   **Package Manager:** uv (Ultra-fast Python package installer)
*   **Background Tasks:** Celery + RabbitMQ (Broker) + PostgreSQL (Result Backend)
*   **Database:** PostgreSQL + [`pgvector`](https://github.com/pgvector/pgvector) extension (Cosine Similarity matching)
*   **Object Storage:** Storage Bucket (S3 Compatible)
*   **Machine Learning:** ONNX runtime (CUDA 11.8), InsightFace
*   **Containerization:** Docker & Docker Compose

---

## ⚙️ How to Deploy Locally

Eventsnap is completely Dockerized for rapid development and testing.

### 1. Build the Docker Images
Because the Python environment for facial recognition (CUDA, ONNX, OpenCV) is massive, we build the images independently to cache the layers effectively before orchestrating them.

```bash
# Build the stateless inference API (requires Nvidia Container Toolkit)
docker build -t inference_api:dev ./inference_api

# Build the FastAPI orchestrator and Celery worker
docker build -t main_api:dev ./main_api
```

### 2. Spin Up the Stack
Bring up all the containers (Postgres DB, RabbitMQ, Storage Bucket, Inference API, Main API, and Celery Worker). The orchestrated services will automatically wait for their database dependencies to become healthy before starting.

```bash
docker compose up -d
```

### 3. Monitor Your Cluster
*   **Main API Orchestrator:** http://localhost:8000/docs
*   **Inference Model API:** http://localhost:5000/docs

---

## 📁 Architecture Flow

*(For detailed sequence diagrams of the complete asynchronous system, see [workflows.md](./workflows.md))*

1.  A user uploads a ZIP of an event directly via the Next.js frontend, which extracts and pushes the images into **Storage Bucket**.
2.  The frontend hits the **Main API** `/encode-event/` endpoint, passing the `event_code` in the JSON payload.
3.  The **Main API** creates a Celery Task and immediately returns a `task_id` so the user isn't stuck waiting.
4.  The background **Celery Worker** picks up the task, pre-fetches images from **Storage Bucket** using an aggressive 64-connection pool, beams them (Base64) to the **Inference API**, and bulk-inserts the generated 512D vectors directly into **PostgreSQL**.
5.  An attendee hits the **Main API** `/sort-attendee/` endpoint with their selfies and the `event_code`. The orchestrator gets the embeddings for those selfies, averages them, and executes a sub-millisecond `<=>` cosine similarity search in `pgvector` to find all photos they appear in!
