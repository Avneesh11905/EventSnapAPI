from dataclasses import dataclass
from uuid import UUID


@dataclass
class EventEncodingDTO:
    id: UUID
    event_code: str
    image_path: str
    embedding: list[float]
    confidence: float


@dataclass
class EncodingStatusDTO:
    task_id: str
    status: str
    progress: str | None = None
    images_processed: int | None = None
    total_images: int | None = None
    message: str | None = None


@dataclass
class EncodedCountDTO:
    encoded_count: int
    table_exists: bool


@dataclass
class DeleteDataDTO:
    success: bool
    message: str | None = None
    table_name: str | None = None


@dataclass
class AttendeeSortDTO:
    event_code: str
    matches_found: int
    photos: list[str]


@dataclass
class ZipCheckDTO:
    exists: bool
    zip_path: str | None = None
    filename: str | None = None


@dataclass
class BackgroundEncodingResult:
    total: int
    skipped: int = 0
    encoded: int = 0
    no_encodings_found: int = 0
    group_id: str | None = None


@dataclass
class BackgroundZipResult:
    zip_path: str
    filename: str
    images_zipped: int


@dataclass
class TaskStatusDTO:
    state: str
    info: dict | None = None
    result: dict | None = None
