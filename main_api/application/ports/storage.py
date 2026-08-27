from typing import Protocol


class IStorageService[T](Protocol):
    async def list_images(self, folder_path: str) -> list[str]:
        pass

    async def download_images(self, keys: list[str]) -> list[T | Exception]:
        pass

    async def create_zip_from_images(
        self, zip_path: str, image_paths: list[dict], progress_callback=None
    ) -> None:
        """
        Creates a zip file in the storage layer.
        progress_callback can be called with (current, total) to report progress.
        """
        pass

    async def check_zip_exists(self, zip_key: str) -> bool:
        pass

    async def delete_folder(self, prefix: str) -> None:
        pass

    async def delete_objects(self, keys: list[str]) -> None:
        pass
