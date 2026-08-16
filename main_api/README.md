# Eventsnap Main API (Orchestrator)

The `main_api` is the high-performance Eventsnap orchestrator built on FastAPI. It handles API authentication, HTTP request routing, database querying (with `pgvector`), and delegating heavy facial encoding workloads to asynchronous Celery workers.

## Core Responsibilities
*   **Async Orchestration**: Rapidly accepts encoding requests and pushes them to RabbitMQ (with Redis as a result backend).
*   **Similarity Search**: Uses PostgreSQL `pgvector` for lightning-fast cosine similarity matching.
*   **Network Optimization**: Automatically compresses large JSON responses using `GZipMiddleware` to minimize bandwidth and accelerate client downloads.

---

## 🏗 Directory Structure (Hexagonal Architecture)

This API strictly follows the **Ports and Adapters (Hexagonal) Architecture** using Dependency Injection (`di_container.py`).

*   **`domain/`**: Enterprise logic, entities, and domain exceptions. Completely independent of any external frameworks.
*   **`application/`**: Business use cases (e.g., `events.py`, `attendees.py`, `background_tasks.py`), DTOs, and Ports (interfaces).
*   **`infrastructure/`**: External adapters (Celery, PostgreSQL/pgvector, MinIO, HuggingFace Inference API, Image Augmentation).
*   **`presentation/`**: FastAPI routers, schemas, and exception handlers.

---

## REST Endpoints

All endpoints require the following Header:
```http
Content-Type: application/json
```

### 1. Encode Event (Asynchronous)
Triggers the background Celery worker to download a folder from MinIO, process every image through the GPU inference API, and save the 512-dimension vector embeddings to Postgres.

**Example `curl` Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/events/encode-event/' \
  -H 'Content-Type: application/json' \
  -d '{
  "event_code": "DCAYTI",
  "max_faces": 0,
  "det_conf": 0.5,
  "nms_thresh": 0.4
}'
```

**Example `axios` (Next.js) Request:**
```javascript
import axios from 'axios';

const response = await axios.post('http://localhost:8000/api/events/encode-event/', {
  event_code: 'DCAYTI',
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
**Response (202 Accepted):**
```json
{
  "message": "Event encoding task has been enqueued to RabbitMQ Worker.",
  "task_id": "74e9fe48-3dda-4b76-8d87-052ffbfa4cec",
  "event_code": "DCAYTI"
}
```

### 2. Check Encoding Status
Retrieves the real-time progress of a running `encode-event` task from the Celery worker via the persistent database result backend.

**Example `curl` Request:**
```bash
curl -X 'GET' \
  'http://localhost:8000/api/events/encode-status/{task_id}'
```

**Example `axios` (Next.js) Request:**
```javascript
import axios from 'axios';

const taskId = '74e9fe48-3dda-4b76...';
const response = await axios.get(`http://localhost:8000/api/events/encode-status/${taskId}`);
console.log(response.data);
```

**Response (In-Progress):**
```json
{
  "task_id": "74e9fe48-3dda-4b76...",
  "status": "PROCESSING",
  "progress": "45%",
  "images_processed": 450,
  "total_images": 1000
}
```

**Response (Complete):**
```json
{
  "task_id": "74e9fe48-3dda-4b76...",
  "status": "SUCCESS",
  "message": "Success"
}
```

### 3. Encode Attendee (High-Precision Augmentation)
Converts 3 raw attendee profile photos (front, left, right) into 9 augmented facial embeddings, ready for database sorting. Runs on a non-blocking asyncio thread pool.

**Example `curl` Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/attendees/encode-attendee/' \
  -H 'Content-Type: application/json' \
  -d '{
  "attendee_images_base64": [
    "base64_string_1...",
    "base64_string_2...",
    "base64_string_3..."
  ]
}'
```

**Example `axios` (Next.js) Request:**
```javascript
import axios from 'axios';

const response = await axios.post('http://localhost:8000/api/attendees/encode-attendee/', {
  attendee_images_base64: [
    'base64_string_1...',
    'base64_string_2...',
    'base64_string_3...'
  ]
}, {
  headers: {
    'Content-Type': 'application/json'
  }
});
console.log(response.data.encodings); // Array of 9 embeddings
```

**Response (200 OK):**
```json
{
  "message": "Successfully generated 9 encodings from 3 reference images.",
  "encodings": [
    [0.012, -0.045, ...], 
    [0.018, -0.052, ...],
    ...
  ]
}
```

### 4. Sort Attendee (High-Speed Matcher)
The core feature: Finds an attendee in an encoded event. It accepts the 9 precise encodings generated in the previous step, and executes a heavily optimized Postgres `pgvector` K-NN (K-Nearest Neighbors) matching query to confidently find the attendee in massive crowd datasets.

**Example `curl` Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/attendees/sort-attendee/' \
  -H 'Content-Type: application/json' \
  -d '{
  "event_code": "DCAYTI",
  "attendee_encodings": [
    [0.012, -0.045, ...],
    [0.018, -0.052, ...]
  ]
}'
```

**Example `axios` (Next.js) Request:**
```javascript
import axios from 'axios';

// Encodings from the previous '/encode-attendee/' step
const encodingsArray = [[...], [...], ...]; 

const response = await axios.post('http://localhost:8000/api/attendees/sort-attendee/', {
  event_code: 'DCAYTI',
  attendee_encodings: encodingsArray
}, {
  headers: {
    'Content-Type': 'application/json'
  }
});
console.log(response.data);
```

**Response (200 OK):**
```json
{
  "event_code": "DCAYTI",
  "matches_found": 14,
  "photos": [
    "events/DCAYTI/DSC_001.jpg",
    "events/DCAYTI/DSC_045.jpg"
  ]
}
```

## Running the Service
The service is booted automatically via `docker-compose up`, but can be tested locally using:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
