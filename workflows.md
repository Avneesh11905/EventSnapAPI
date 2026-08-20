# Eventsnap Full-Stack Workflows

This document contains Mermaid sequence diagrams detailing the asynchronous data flow across the entire Eventsnap architecture, spanning from the Next.js frontend to the FastAPI backend, Celery workers, PostgreSQL databases, and Storage Buckets.

## 1. Event Creation & Photo Upload (Organizer)

When an organizer creates an event and uploads a massive folder of photos, the files are uploaded directly to the Storage Bucket using pre-signed URLs to reduce server load. The AI encoding is then offloaded to background Celery workers so the frontend is never blocked.

```mermaid
sequenceDiagram
    actor Organizer as Organizer / Next.js Client
    participant NextAPI as Next.js API (BFF)
    participant Prisma as PostgreSQL (Prisma)
    participant FastAPI as Main API (FastAPI)
    participant RMQ as RabbitMQ (Broker)
    participant Worker as Celery Worker
    participant Storage as Storage Bucket
    participant GPU as Inference API (GPU)
    participant PGVector as PostgreSQL (pgvector)

    Organizer->>NextAPI: 1. Create Event
    NextAPI->>Prisma: 2. Save Event Metadata
    NextAPI-->>Organizer: 3. Return Event Code

    Organizer->>NextAPI: 4. Request Pre-signed URLs for upload
    NextAPI-->>Organizer: 5. Return URLs
    Organizer->>Storage: 6. Upload raw & thumb photos (event/{code}/raw/ & /thumbs/)

    Organizer->>NextAPI: 7. Click "Start Recognition" (/api/encode)
    NextAPI->>FastAPI: 8. POST /api/events/encode-event/
    FastAPI->>RMQ: 9. Enqueue Encode Task
    FastAPI-->>NextAPI: 10. Returns task_id
    NextAPI-->>Organizer: 11. Returns task_id

    par Background Encoding
        RMQ->>Worker: 12. Pick up Master Task
        loop Every Batch of 64 Images
            Worker->>Storage: 13. Fetch thumb photos (/thumbs/)
            Storage-->>Worker: Photos
            Worker->>GPU: 14. POST base64 images
            GPU-->>Worker: 15. Return 512D Embeddings & BBoxes
            Worker->>PGVector: 16. Bulk Insert encodings
            Worker->>PGVector: 17. Update Master Task Progress State
        end
    and Client Polling
        loop Every 2 Seconds
            Organizer->>NextAPI: 18. Poll /api/upload/status?taskId=...
            NextAPI->>FastAPI: 19. GET /api/events/encode-status/{task_id}
            FastAPI->>PGVector: Check Celery task state
            FastAPI-->>Organizer: Returns Progress (e.g. 45%)
        end
    end
```

## 2. Registration & Face Encoding (Attendee)

When an attendee registers, they use their webcam to capture 3 selfies. The 512D face embeddings are stored directly in their NextAuth user profile via Prisma, so they only ever have to register their face once.

```mermaid
sequenceDiagram
    actor Attendee as Attendee / Next.js Client
    participant NextAPI as Next.js API (BFF)
    participant FastAPI as Main API (FastAPI)
    participant GPU as Inference API (GPU)
    participant Prisma as PostgreSQL (Prisma)

    Attendee->>NextAPI: 1. Submit 3 Selfies (Front, Left, Right)
    NextAPI->>FastAPI: 2. POST /api/attendees/encode-attendee/
    Note over FastAPI: Image Augmentation<br/>(Flips, Rotation, Contrast)
    FastAPI->>GPU: 3. POST 9 augmented base64 images
    GPU-->>FastAPI: 4. Return 9 precise embeddings
    FastAPI-->>NextAPI: 5. Returns 9 embeddings array

    NextAPI->>Prisma: 6. Save embeddings array to User record
    NextAPI-->>Attendee: 7. Update NextAuth session (hasEncoding=true)
```

## 3. Event Scanning & Matching (Attendee)

When an attendee wants to find their photos, the backend leverages pgvector to perform a lightning-fast K-Nearest Neighbors (K-NN) cosine similarity search against the millions of faces found in the event.

```mermaid
sequenceDiagram
    actor Attendee as Attendee / Next.js Client
    participant NextAPI as Next.js API (BFF)
    participant Prisma as PostgreSQL (Prisma)
    participant FastAPI as Main API (FastAPI)
    participant PGVector as PostgreSQL (pgvector)
    participant Storage as Storage Bucket

    Attendee->>NextAPI: 1. Enter Event Code
    NextAPI->>Prisma: 2. Fetch Attendee's saved embeddings
    NextAPI->>FastAPI: 3. POST /api/attendees/sort-attendee/ (code, embeddings)
    Note over FastAPI: Averages the 9 embeddings<br/>into 1 highly accurate vector
    
    FastAPI->>PGVector: 4. K-NN Cosine Similarity (<=>)
    PGVector-->>FastAPI: 5. Returns Matched S3 Keys
    FastAPI-->>NextAPI: 6. Returns Matched Keys
    
    NextAPI->>Storage: 7. Generate pre-signed GET URLs for thumb keys
    NextAPI->>Prisma: 8. Cache event access record
    NextAPI-->>Attendee: 9. Returns Photos Array & redirects to /events/[id]
```

## 4. ZIP Generation & Download (Attendee)

Attendees can download all their matched photos as a ZIP file. Because compressing hundreds of high-res photos is computationally heavy and slow, this is handled asynchronously by the Celery Worker.

```mermaid
sequenceDiagram
    actor Attendee as Attendee / Next.js Client
    participant NextAPI as Next.js API (BFF)
    participant FastAPI as Main API (FastAPI)
    participant RMQ as RabbitMQ (Broker)
    participant Worker as Celery Worker
    participant Storage as Storage Bucket

    Attendee->>NextAPI: 1. Click "Generate ZIP"
    NextAPI->>FastAPI: 2. POST /api/attendees/generate-zip/ (event_id, keys)
    FastAPI->>RMQ: 3. Enqueue Zip Task
    FastAPI-->>NextAPI: 4. Returns task_id
    NextAPI-->>Attendee: 5. Returns task_id

    par Background Compression
        RMQ->>Worker: 6. Pick up ZIP Task
        Worker->>Storage: 7. Fetch matched raw photos
        Storage-->>Worker: Photos
        Note over Worker: Compresses raw photos into .zip
        Worker->>Storage: 8. Upload .zip (zip/{event_id}/{user_id}.zip)
        Worker->>Worker: 9. Mark task SUCCESS
    and Client Polling
        loop Polling
            Attendee->>NextAPI: 10. Poll /api/tasks/{taskId}
            NextAPI->>FastAPI: 11. Check Task Status
            FastAPI-->>Attendee: Returns status
        end
    end
    
    Attendee->>NextAPI: 12. On Success: GET /api/attendee/check-zip
    NextAPI->>Storage: 13. Generate pre-signed Download URL
    NextAPI-->>Attendee: 14. Returns Download URL
```

## 5. Event Deletion & Cleanup (Organizer)

To prevent Next.js serverless timeouts when deleting an event with thousands of photos and embeddings, the heavy cleanup is delegated to the Python backend.

```mermaid
sequenceDiagram
    actor Organizer as Organizer / Next.js Client
    participant NextAPI as Next.js API (BFF)
    participant Prisma as PostgreSQL (Prisma)
    participant FastAPI as Main API (FastAPI)
    participant RMQ as RabbitMQ (Broker)
    participant Worker as Celery Worker
    participant PGVector as PostgreSQL (pgvector)
    participant Storage as Storage Bucket

    Organizer->>NextAPI: 1. Click "Delete Event"
    NextAPI->>Prisma: 2. Delete event metadata
    NextAPI->>FastAPI: 3. DELETE /api/events/delete-event-table/{code}?event_id={id}
    FastAPI->>RMQ: 4. Enqueue Cleanup Task
    FastAPI-->>NextAPI: 5. Returns instantly (fire-and-forget)
    NextAPI-->>Organizer: 6. Returns 200 OK (Event disappears from UI)

    RMQ->>Worker: 7. Pick up Cleanup Task
    Worker->>PGVector: 8. DELETE FROM event_encodings WHERE event_code = ...
    Worker->>Storage: 9. Recursively delete folder event/{code}/
    Worker->>Storage: 10. Recursively delete folder zip/{event_id}/
```
