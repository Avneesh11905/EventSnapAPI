from typing import List, Set, Dict, Any, Protocol

class EventRepository(Protocol):
    async def create_event_table(self, folder_path: str) -> None:
        pass

    async def get_already_encoded_images(self, folder_path: str) -> Set[str]:
        pass

    async def save_encodings(self, folder_path: str, encodings: List[Dict[str, Any]]) -> None:
        pass

    async def check_table_exists(self, folder_path: str) -> bool:
        pass

    async def get_encoded_count(self, folder_path: str) -> int:
        pass

    async def delete_event_table(self, folder_path: str) -> None:
        pass

    async def find_matches(self, folder_path: str, encodings: List[List[float]], threshold: float, min_matches: int) -> List[str]:
        pass

    async def get_closest_matches_debug(self, folder_path: str, encodings: List[List[float]], limit: int = 5) -> List[Dict[str, Any]]:
        pass
