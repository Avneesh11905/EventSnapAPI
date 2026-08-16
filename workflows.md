# Eventsnap Workflows

This document contains Mermaid sequence diagrams detailing the asynchronous data flow across the Eventsnap Hexagonal Architecture.

## 1. Event Encoding Workflow (Background Processing)

When a photographer uploads a massive event folder, the Main API delegates the heavy lifting to the background Celery workers so the frontend is never blocked.

```mermaid
sequenceDiagram
    actor User as Photographer / Next.js
    participant API as Main API (FastAPI)
    participant RMQ as RabbitMQ (Broker)
    participant Redis as Redis (Result Backend)
    participant Worker as Celery Worker
    participant MinIO as MinIO Storage
    participant GPU as Inference API (GPU)
    participant DB as PostgreSQL (pgvector)

    User->>MinIO: 1. Upload ZIP / Photos directly
    User->>API: 2. POST /api/encode-event/ (event_code)
    API->>RMQ: 3. Enqueue Encode Task
    API-->>User: 4. Returns 202 Accepted (task_id)

    RMQ->>Worker: 5. Pick up Task

    loop Every Batch of Images
        Worker->>MinIO: 6. Fetch raw photos
        MinIO-->>Worker: Photos
        Worker->>GPU: 7. POST base64 images
        GPU-->>Worker: 8. Return 512D Embeddings & BBoxes
        Worker->>DB: 9. Bulk Insert into pgvector
        Worker->>Redis: 10. Update Progress State
    end

    loop Polling
        User->>API: 11. GET /api/encode-status/{task_id}
        API->>Redis: Check status
        API-->>User: Returns Progress (e.g. 45%)
    end
```

## 2. Attendee Sorting Workflow (Lightning Fast Matching)

When an attendee wants to find their photos, the Main API uses Data Augmentation and highly-optimized pgvector queries to match them in milliseconds.

```mermaid
sequenceDiagram

actor Attendee as Attendee / Next.js
participant API as Main API (FastAPI)
participant GPU as Inference API (GPU)
participant DB as PostgreSQL (pgvector)

Attendee->>API: 1. POST /api/encode-attendee/ (3 selfies)
Note over API: Image Augmentation<br/>(Flips, Rotation, Contrast)
API->>GPU: 2. POST 9 augmented base64 images
GPU-->>API: 3. Return 9 precise embeddings
API-->>Attendee: 4. Returns embeddings array

Attendee->>API: 5. POST /api/sort-attendee/ (event_code, embeddings)
Note over API: Averages the 9 embeddings<br/>into 1 highly accurate vector
API->>DB: 6. pgvector K-NN Cosine Similarity (<=>)
DB-->>API: 7. Returns Matched Image URLs
API-->>Attendee: 8. Returns Photos Array
```

## 3. ZIP Generation Workflow

Attendees can download all their matched photos as a ZIP file. This is also handled asynchronously by the Celery Worker.

```mermaid
sequenceDiagram

actor Attendee as Attendee / Next.js
participant API as Main API (FastAPI)
participant RMQ as RabbitMQ (Broker)
participant Worker as Celery Worker
participant MinIO as MinIO Storage

Attendee->>API: 1. POST /api/generate-zip/ (event_id, user_id, photos)
API->>RMQ: 2. Enqueue Zip Task
API-->>Attendee: 3. Returns 202 Accepted (task_id)

RMQ->>Worker: 4. Pick up Task
Worker->>MinIO: 5. Fetch all matched photos
MinIO-->>Worker: Photos
Note over Worker: Compresses into .zip
Worker->>MinIO: 6. Upload .zip file

loop Polling
    Attendee->>API: 7. GET /api/check-zip/{event_id}/{user_id}
    API->>MinIO: Check if ZIP exists
    API-->>Attendee: Returns true/false & Download URL
end
```
