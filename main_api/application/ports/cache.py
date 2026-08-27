from typing import Protocol


class ICacheService(Protocol):
    async def acquire_lock(self, lock_name: str, expiration: int) -> bool:
        """Attempts to acquire a lock. Returns True if successful, False otherwise."""

    async def release_lock(self, lock_name: str) -> None:
        """Releases the lock."""

    async def set_flag(self, key: str, expiration: int) -> None:
        """Sets a boolean flag with an expiration."""

    async def get_flag(self, key: str) -> bool:
        """Gets a boolean flag."""
