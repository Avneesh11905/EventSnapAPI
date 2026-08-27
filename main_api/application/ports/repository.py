from typing import Protocol

from application.dtos import EventEncodingDTO


class IEventRepository(Protocol):
    async def get_already_encoded_images(self, event_code: str) -> set[str]:
        pass

    async def save_encodings(self, encodings: list[EventEncodingDTO]) -> None:
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
        encoding: list[float],
        threshold: float,
    ) -> list[str]:
        pass

    async def save_processed_images(self, processed_data: list[dict]) -> None:
        pass

    async def delete_keys(self, event_code: str, keys: list[str]) -> None:
        pass

    async def get_image_status(self, event_code: str) -> dict:
        pass
