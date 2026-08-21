from application.ports.repository import IEventRepository
from infrastructure.database.models import EventEncodingModel
from application.dtos import EventEncodingDTO
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import (
    insert,
    select,
    delete,
    func,
    literal_column,
    Float,
)
from typing import List, Set
import dataclasses


class PostgresEventRepository(IEventRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_already_encoded_images(self, event_code: str) -> Set[str]:
        stmt = (
            select(EventEncodingModel.image_path)
            .where(EventEncodingModel.event_code == event_code)
            .distinct()
        )
        result = await self.session.execute(stmt)
        return {row[0] for row in result.fetchall()}

    async def save_encodings(self, encodings: List[EventEncodingDTO]) -> None:
        encoding_dicts = [dataclasses.asdict(e) for e in encodings]
        await self.session.execute(insert(EventEncodingModel), encoding_dicts)

    async def check_event_has_data(self, event_code: str) -> bool:
        stmt = (
            select(1)
            .select_from(EventEncodingModel)
            .where(EventEncodingModel.event_code == event_code)
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar() is not None

    async def get_encoded_count(self, event_code: str) -> int:
        stmt = select(func.count(EventEncodingModel.image_path.distinct())).where(
            EventEncodingModel.event_code == event_code
        )
        result = await self.session.execute(stmt)
        count = result.scalar() or 0
        return count

    async def delete_event_data(self, event_code: str) -> None:
        stmt = delete(EventEncodingModel).where(
            EventEncodingModel.event_code == event_code
        )
        await self.session.execute(stmt)

    async def find_matches(
        self,
        event_code: str,
        encoding: List[float],
        threshold: float,
    ) -> List[str]:
        distance_op = EventEncodingModel.embedding.op("<=>", return_type=Float())(
            encoding
        )

        stmt = (
            select(
                EventEncodingModel.image_path,
                func.min(distance_op).label("best_distance"),
            )
            .where(EventEncodingModel.event_code == event_code, distance_op < threshold)
            .group_by(EventEncodingModel.image_path)
            .order_by(literal_column("best_distance").asc())
        )

        result = await self.session.execute(stmt)
        rows = result.all()
        return [row[0] for row in rows]
