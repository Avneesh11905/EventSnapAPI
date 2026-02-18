# Eventsnap Storage API

The `storage_api` is a lightweight, asynchronous FastAPI microservice explicitly designed to act as a bridge between the Next.js Frontend and the MinIO/S3 Storage backend. 

> **Important:** This service is built strictly for **testing purposes** and is not intended for real-world production use.

Currently, it handles complex file streaming operations such as extracting `.zip` test archives directly into Cloud Storage.

## Core Responsibilities
*   **Unzipping**: Streams uploaded ZIP archives directly into S3 buckets instead of blocking the `main_api` event loop.
*   **S3 Abstraction**: Utilizes asynchronous `aioboto3` to perform high-speed I/O.

---

## REST Endpoints

### 1. Upload Event ZIP Archive
Accepts a generic `.zip` file of an event via multipart form-data, safely streams it from memory to disk, extracts its contents asynchronously, and pushes all images (ignoring system files like `.DS_Store`) straight into a MinIO bucket folder.

**Request:** `POST /upload-event/`
*Multipart Form Data:*
*   `file`: The `.zip` binary archive containing event photos.
*   `event_name`: A string identifier for the folder (e.g., `summer_fest_2026`).

**Example `curl` Request:**
```bash
curl -X 'POST' \
  'http://localhost:8001/upload-event/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@my_photos.zip;type=application/zip' \
  -F 'event_name=summer_fest_2026'
```

**Example `axios` (Next.js) Request:**
```javascript
import axios from 'axios';

const formData = new FormData();
// Assuming `fileInput` is an HTML input element: <input type="file" id="fileInput" />
formData.append('file', fileInput.files[0]);
formData.append('event_name', 'summer_fest_2026');

const response = await axios.post('http://localhost:8001/upload-event/', formData, {
  headers: {
    'accept': 'application/json',
    'Content-Type': 'multipart/form-data'
  }
});
console.log(response.data);
```

**Response (200 OK):**
```json
{
  "message": "Successfully uploaded and extracted 45 files to events/summer_fest_2026",
  "folder_path": "events/summer_fest_2026"
}
```

*Note: Once this endpoint returns the `folder_path`, the Next.js frontend traditionally passes that exact path into the `main_api`'s `/encode-event/` endpoint to start the GPU pipeline!*

## Running the Service
The standalone storage API is built around `aioboto3`. To run it locally:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
