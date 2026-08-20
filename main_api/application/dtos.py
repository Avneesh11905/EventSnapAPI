from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID


@dataclass
class EventEncodingDTO:
    id: UUID
    event_code: str
    image_path: str
    embedding: List[float]
    confidence: float


@dataclass
class EncodingStatusDTO:
    task_id: str
    status: str
    progress: Optional[str] = None
    images_processed: Optional[int] = None
    total_images: Optional[int] = None
    message: Optional[str] = None


@dataclass
class EncodedCountDTO:
    encoded_count: int
    table_exists: bool


@dataclass
class DeleteDataDTO:
    success: bool
    message: Optional[str] = None
    table_name: Optional[str] = None


@dataclass
class AttendeeSortDTO:
    event_code: str
    matches_found: int
    photos: List[str]


@dataclass
class ZipCheckDTO:
    exists: bool
    zip_path: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class BackgroundEncodingResult:
    result: str
    total: int
    skipped: int = 0


@dataclass
class BackgroundZipResult:
    status: str
    progress: int
    zip_path: str
    filename: str
