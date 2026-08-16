from application.ports.uow import IUnitOfWork
from infrastructure.database.repository import PostgresEventRepository
from sqlalchemy.ext.asyncio import async_sessionmaker


class AsyncSqlAlchemyUnitOfWork(IUnitOfWork):
    def __init__(self, session_factory: async_sessionmaker):
        self.session_factory = session_factory

    async def __aenter__(self):
        self.session = self.session_factory()
        self.event_repo = PostgresEventRepository(self.session)
        return self

    async def __aexit__(self, exc_type, exc_val, traceback):
        if exc_type is not None:
            await self.rollback()
        else:
            # We don't auto-commit here, the user should explicitly call commit()
            # if they want to commit changes. Auto-commit on exit can be dangerous.
            # But wait, standard UoW often auto-commits if no exception?
            # Actually, explicit commit is safer, so if they didn't commit, it rolls back implicitly.
            pass
        await self.session.close()

    async def commit(self):
        await self.session.commit()

    async def rollback(self):
        await self.session.rollback()
