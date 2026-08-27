import valkey.asyncio as valkey

from application.ports.cache import ICacheService


class ValkeyCacheService(ICacheService):
    def __init__(self, valkey_url: str) -> None:
        self.client = valkey.from_url(valkey_url)

    async def acquire_lock(self, lock_name: str, expiration: int) -> bool:
        lock_acquired = await self.client.set(lock_name, "1", nx=True, ex=expiration)
        return bool(lock_acquired)

    async def release_lock(self, lock_name: str) -> None:
        await self.client.delete(lock_name)

    async def set_flag(self, key: str, expiration: int) -> None:
        await self.client.set(key, "1", ex=expiration)

    async def get_flag(self, key: str) -> bool:
        val = await self.client.get(key)
        return val is not None
