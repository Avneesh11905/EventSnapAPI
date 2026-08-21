from pydantic import BaseModel
from typing import List, Optional

# ---- Request Models ----


class EncodeEventRequest(BaseModel):
    event_code: str
    detection_conf: float = 0.5
    nms_threshold: float = 0.4


class EncodeAttendeeRequest(BaseModel):
    attendee_images_base64: List[str]


class SortAttendeeRequest(BaseModel):
    event_code: str
    attendee_encoding: List[float]


class GenerateZipRequest(BaseModel):
    event_id: str
    user_id: str
    image_paths: List[dict]


# ---- API Only Response Models ----


class HealthResponse(BaseModel):
    status: str
    service: str
    checks: dict


class EnqueueTaskResponse(BaseModel):
    message: str
    task_id: str
    event_code: Optional[str] = None


class EncodeAttendeeResponse(BaseModel):
    message: str
    encoding: List[float]


class GenerateZipResponse(BaseModel):
    success: bool
    task_id: str
    message: str


class EncodingStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[str] = None
    images_processed: Optional[int] = None
    total_images: Optional[int] = None
    message: Optional[str] = None


class EncodedCountResponse(BaseModel):
    encoded_count: int
    table_exists: bool


class DeleteDataResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    table_name: Optional[str] = None


class AttendeeSortResponse(BaseModel):
    event_code: str
    matches_found: int
    photos: List[str]


class ZipCheckResponse(BaseModel):
    exists: bool
    zip_path: Optional[str] = None
    filename: Optional[str] = None
