from typing import List, Set, Protocol
from application.dtos import EventEncodingDTO


class IEventRepository(Protocol):
    async def get_already_encoded_images(self, event_code: str) -> Set[str]:
        pass

    async def save_encodings(self, encodings: List[EventEncodingDTO]) -> None:
        pass

    async def check_event_has_data(self, event_code: str) -> bool:
        pass

    async def get_encoded_count(self, event_code: str) -> int:
        pass

    async def delete_event_data(self, event_code: str) -> None:
        pass

    async def find_matches(
        self,
        event_code: str,
        encoding: List[float],
        threshold: float,
    ) -> List[str]:
        pass

    async def save_processed_images(self, processed_data: List[dict]) -> None:
        pass

    async def delete_keys(self, event_code: str, keys: List[str]) -> None:
        pass

    async def get_image_status(self, event_code: str) -> dict:
        pass
