from typing import Protocol, Any
from .repository import IEventRepository


class IUnitOfWork(Protocol):
    event_repo: IEventRepository

    async def __aenter__(self) -> "IUnitOfWork":
        pass

    async def __aexit__(self, exc_type: Any, exc_val: Any, traceback: Any) -> None:
        pass

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass
