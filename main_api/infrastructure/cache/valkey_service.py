from application.ports.cache import ICacheService
import valkey.asyncio as valkey


class ValkeyCacheService(ICacheService):
    def __init__(self, valkey_url: str):
        self.client = valkey.from_url(valkey_url)

    async def acquire_lock(self, lock_name: str, expiration: int) -> bool:
        lock_acquired = await self.client.set(lock_name, "1", nx=True, ex=expiration)
        return bool(lock_acquired)

    async def release_lock(self, lock_name: str) -> None:
        await self.client.delete(lock_name)
