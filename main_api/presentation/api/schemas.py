from pydantic import BaseModel

# ---- Request Models ----


class EncodeEventRequest(BaseModel):
    event_code: str
    detection_conf: float = 0.5
    nms_threshold: float = 0.4


class EncodeAttendeeRequest(BaseModel):
    attendee_images_base64: list[str]


class SortAttendeeRequest(BaseModel):
    event_code: str
    attendee_encoding: list[float]


class GenerateZipRequest(BaseModel):
    event_id: str
    user_id: str
    image_paths: list[dict]


# ---- API Only Response Models ----


class HealthResponse(BaseModel):
    status: str
    service: str
    checks: dict


class EnqueueTaskResponse(BaseModel):
    message: str
    task_id: str
    event_code: str | None = None


class EncodeAttendeeResponse(BaseModel):
    message: str
    encoding: list[float]


class GenerateZipResponse(BaseModel):
    success: bool
    task_id: str
    message: str


class EncodingStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: str | None = None
    images_processed: int | None = None
    total_images: int | None = None
    message: str | None = None


class EncodedCountResponse(BaseModel):
    encoded_count: int
    table_exists: bool


class DeleteDataResponse(BaseModel):
    success: bool
    message: str | None = None
    table_name: str | None = None


class AttendeeSortResponse(BaseModel):
    event_code: str
    matches_found: int
    photos: list[str]


class ZipCheckResponse(BaseModel):
    exists: bool
    zip_path: str | None = None
    filename: str | None = None


class TaskStatusResponse(BaseModel):
    state: str
    info: dict | None = None
    result: dict | None = None
