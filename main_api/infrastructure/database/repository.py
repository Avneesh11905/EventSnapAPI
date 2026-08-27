from application.ports.repository import IEventRepository
from infrastructure.database.models import EventEncodingModel, ProcessedImageModel
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
            select(ProcessedImageModel.image_path)
            .where(ProcessedImageModel.event_code == event_code)
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

        stmt_processed = delete(ProcessedImageModel).where(
            ProcessedImageModel.event_code == event_code
        )
        await self.session.execute(stmt_processed)

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

    async def save_processed_images(self, processed_data: List[dict]) -> None:
        if not processed_data:
            return
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(ProcessedImageModel).values(processed_data)
        stmt = stmt.on_conflict_do_nothing(index_elements=["event_code", "image_path"])
        await self.session.execute(stmt)

    async def delete_keys(self, event_code: str, keys: List[str]) -> None:
        if not keys:
            return
        stmt1 = delete(EventEncodingModel).where(
            EventEncodingModel.event_code == event_code,
            EventEncodingModel.image_path.in_(keys),
        )
        stmt2 = delete(ProcessedImageModel).where(
            ProcessedImageModel.event_code == event_code,
            ProcessedImageModel.image_path.in_(keys),
        )
        await self.session.execute(stmt1)
        await self.session.execute(stmt2)

    async def get_image_status(self, event_code: str) -> dict:
        stmt_processed = (
            select(ProcessedImageModel.image_path)
            .where(ProcessedImageModel.event_code == event_code)
            .distinct()
        )
        stmt_faces = (
            select(EventEncodingModel.image_path)
            .where(EventEncodingModel.event_code == event_code)
            .distinct()
        )

        processed_result = await self.session.execute(stmt_processed)
        faces_result = await self.session.execute(stmt_faces)

        processed_set = {row[0] for row in processed_result.fetchall()}
        faces_set = {row[0] for row in faces_result.fetchall()}

        no_faces_set = processed_set - faces_set

        return {"has_faces": list(faces_set), "no_faces": list(no_faces_set)}
