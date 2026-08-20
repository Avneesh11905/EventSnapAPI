# Eventsnap Workflows

This document contains Mermaid sequence diagrams detailing the asynchronous data flow across the Eventsnap Hexagonal Architecture.

## 1. Event Encoding Workflow (Background Processing)

When a photographer uploads a massive event folder, the Main API delegates the heavy lifting to the background Celery workers so the frontend is never blocked.

```mermaid
sequenceDiagram
    actor User as Photographer / Next.js
    participant API as Main API (FastAPI)
    participant RMQ as RabbitMQ (Broker)
    participant Worker as Celery Worker
    participant Storage Bucket as Storage Bucket Storage
    participant GPU as Inference API (GPU)
    participant DB as PostgreSQL (Result Backend & pgvector)

    User->>Storage Bucket: 1. Upload ZIP / Photos directly
    User->>API: 2. POST /api/events/encode-event/ (event_code)
    API->>RMQ: 3. Enqueue Encode Task
    API-->>User: 4. Returns 202 Accepted (task_id)

    RMQ->>Worker: 5. Pick up Task

    loop Every Batch of Images
        Worker->>Storage Bucket: 6. Fetch raw photos
        Storage Bucket-->>Worker: Photos
        Worker->>GPU: 7. POST base64 images
        GPU-->>Worker: 8. Return 512D Embeddings & BBoxes
        Worker->>DB: 9. Bulk Insert into pgvector
        Worker->>DB: 10. Update Progress State
    end

    loop Polling
        User->>API: 11. GET /api/events/encode-status/{task_id}
        API->>DB: Check status
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

Attendee->>API: 1. POST /api/attendees/encode-attendee/ (3 selfies)
Note over API: Image Augmentation<br/>(Flips, Rotation, Contrast)
API->>GPU: 2. POST 9 augmented base64 images
GPU-->>API: 3. Return 9 precise embeddings
API-->>Attendee: 4. Returns embeddings array

Attendee->>API: 5. POST /api/attendees/sort-attendee/ (event_code, embeddings)
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
participant Storage Bucket as Storage Bucket Storage

Attendee->>API: 1. POST /api/attendees/generate-zip/ (event_id, user_id, photos)
API->>RMQ: 2. Enqueue Zip Task
API-->>Attendee: 3. Returns 202 Accepted (task_id)

RMQ->>Worker: 4. Pick up Task
Worker->>Storage Bucket: 5. Fetch all matched photos
Storage Bucket-->>Worker: Photos
Note over Worker: Compresses into .zip
Worker->>Storage Bucket: 6. Upload .zip file

loop Polling
    Attendee->>API: 7. GET /api/attendees/check-zip/{event_id}/{user_id}
    API->>Storage Bucket: Check if ZIP exists
    API-->>Attendee: Returns true/false & Download URL
end
```
