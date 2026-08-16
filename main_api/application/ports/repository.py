from typing import List, Set, Dict, Any, Protocol
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
        encodings: List[List[float]],
        threshold: float,
        min_matches: int,
    ) -> List[str]:
        pass

    async def get_closest_matches_debug(
        self, event_code: str, encodings: List[List[float]], limit: int = 5
    ) -> List[Dict[str, Any]]:
        pass
