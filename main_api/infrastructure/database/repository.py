from application.ports.repository import IEventRepository
from infrastructure.database.models import EventEncodingModel
from application.dtos import EventEncodingDTO
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy import insert, select, delete, func, literal_column, values, column
from pgvector.sqlalchemy import Vector
from typing import List, Set, Dict, Any
import dataclasses


class PostgresEventRepository(IEventRepository):
    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.SessionLocal = async_sessionmaker(
            autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
        )

    async def get_already_encoded_images(self, event_code: str) -> Set[str]:
        async with self.SessionLocal() as db:
            stmt = select(EventEncodingModel.image_path).where(EventEncodingModel.event_code == event_code).distinct()
            result = await db.execute(stmt)
            return {row[0] for row in result.fetchall()}

    async def save_encodings(
        self, encodings: List[EventEncodingDTO]
    ) -> None:
        async with self.SessionLocal() as db:
            encoding_dicts = [dataclasses.asdict(e) for e in encodings]
            await db.execute(insert(EventEncodingModel), encoding_dicts)
            await db.commit()

    async def check_event_has_data(self, event_code: str) -> bool:
        async with self.SessionLocal() as db:
            stmt = select(1).select_from(EventEncodingModel).where(EventEncodingModel.event_code == event_code).limit(1)
            result = await db.execute(stmt)
            return result.scalar() is not None

    async def get_encoded_count(self, event_code: str) -> int:
        async with self.SessionLocal() as db:
            stmt = select(func.count(EventEncodingModel.image_path.distinct())).where(EventEncodingModel.event_code == event_code)
            result = await db.execute(stmt)
            count = result.scalar() or 0
        return count

    async def delete_event_data(self, event_code: str) -> None:
        async with self.SessionLocal() as db:
            stmt = delete(EventEncodingModel).where(EventEncodingModel.event_code == event_code)
            await db.execute(stmt)
            await db.commit()

    async def find_matches(
        self,
        event_code: str,
        encodings: List[List[float]],
        threshold: float,
        min_matches: int,
    ) -> List[str]:
        ref_encodings = values(
            column('id'),
            column('embedding', Vector(512)),
            name="ref_encodings"
        ).data([
            (i + 1, emb) for i, emb in enumerate(encodings)
        ]).cte("ref_encodings")

        distance_op = EventEncodingModel.embedding.op('<=>')(ref_encodings.c.embedding)

        stmt = (
            select(
                EventEncodingModel.image_path,
                func.count(ref_encodings.c.id).label("match_count"),
                func.min(distance_op).label("best_distance")
            )
            .join(ref_encodings, literal_column("true"))
            .where(
                EventEncodingModel.event_code == event_code,
                distance_op < threshold
            )
            .group_by(EventEncodingModel.image_path)
            .having(func.count(ref_encodings.c.id) >= min_matches)
            .order_by(
                literal_column("match_count").desc(),
                literal_column("best_distance").asc()
            )
        )

        async with self.SessionLocal() as db:
            result = await db.execute(stmt)
            rows = result.all()
            return [row[0] for row in rows]

    async def get_closest_matches_debug(
        self, event_code: str, encodings: List[List[float]], limit: int = 5
    ) -> List[Dict[str, Any]]:
        ref_encodings = values(
            column('id'),
            column('embedding', Vector(512)),
            name="ref_encodings"
        ).data([
            (i + 1, emb) for i, emb in enumerate(encodings)
        ]).cte("ref_encodings")

        distance_op = EventEncodingModel.embedding.op('<=>')(ref_encodings.c.embedding)

        stmt = (
            select(
                EventEncodingModel.image_path,
                func.count(ref_encodings.c.id).label("match_count"),
                func.min(distance_op).label("best_distance")
            )
            .join(ref_encodings, literal_column("true"))
            .where(EventEncodingModel.event_code == event_code)
            .group_by(EventEncodingModel.image_path)
            .order_by(literal_column("best_distance").asc())
            .limit(limit)
        )

        async with self.SessionLocal() as db:
            result = await db.execute(stmt)
            return [
                {"image_path": row[0], "match_count": row[1], "best_distance": row[2]}
                for row in result.all()
            ]
