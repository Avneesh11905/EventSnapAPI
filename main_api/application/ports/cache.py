from typing import Protocol


class ICacheService(Protocol):
    async def acquire_lock(self, lock_name: str, expiration: int) -> bool:
        """Attempts to acquire a lock. Returns True if successful, False otherwise."""
        pass

    async def release_lock(self, lock_name: str) -> None:
        """Releases the lock."""
        pass
