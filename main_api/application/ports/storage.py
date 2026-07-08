from typing import List, AsyncGenerator, Protocol

class StorageService(Protocol):
    async def list_images(self, folder_path: str) -> List[str]:
        pass

    async def download_image_b64(self, key: str, max_size: int = 1024) -> str | None:
        pass
    
    async def create_zip_from_images(self, zip_path: str, image_paths: List[dict], progress_callback=None) -> None:
        """
        Creates a zip file in the storage layer.
        progress_callback can be called with (current, total) to report progress.
        """
        pass
    
    async def check_zip_exists(self, zip_key: str) -> bool:
        pass
