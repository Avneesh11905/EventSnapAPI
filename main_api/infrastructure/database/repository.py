from application.ports.repository import EventRepository
from infrastructure.database.models import Base, create_event_table_model, get_event_table_name
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy import text, insert
from typing import List, Set, Dict, Any

class PostgresEventRepository(EventRepository):
    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

    async def create_event_table(self, folder_path: str) -> None:
        create_event_table_model(folder_path)
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)

    async def get_already_encoded_images(self, folder_path: str) -> Set[str]:
        table_name = get_event_table_name(folder_path)
        async with self.SessionLocal() as db:
            result = await db.execute(
                text(f'SELECT DISTINCT image_path FROM "{table_name}"')
            )
            return {row[0] for row in result.fetchall()}

    async def save_encodings(self, folder_path: str, encodings: List[Dict[str, Any]]) -> None:
        EventModel = create_event_table_model(folder_path)
        async with self.SessionLocal() as db:
            await db.execute(insert(EventModel), encodings)
            await db.commit()

    async def check_table_exists(self, folder_path: str) -> bool:
        table_name = get_event_table_name(folder_path)
        from sqlalchemy import inspect
        async with self.engine.connect() as conn:
            def check_table(sync_conn):
                return inspect(sync_conn).has_table(table_name)
            has_table = await conn.run_sync(check_table)
        return has_table

    async def get_encoded_count(self, folder_path: str) -> int:
        table_name = get_event_table_name(folder_path)
        async with self.engine.begin() as conn:
            result = await conn.execute(text(f'SELECT COUNT(DISTINCT image_path) FROM "{table_name}"'))
            count = result.scalar() or 0
        return count

    async def delete_event_table(self, folder_path: str) -> None:
        table_name = get_event_table_name(folder_path)
        async with self.engine.begin() as conn:
            await conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))

    async def find_matches(self, folder_path: str, encodings: List[List[float]], threshold: float, min_matches: int) -> List[str]:
        table_name = get_event_table_name(folder_path)
        formatted_encodings = [f"'{str(emb)}'" for emb in encodings]
        values_clause = ", ".join([f"({i+1}, {emb}::vector)" for i, emb in enumerate(formatted_encodings)])
        
        query_str = f"""
            WITH ref_encodings(id, embedding) AS (
                VALUES {values_clause}
            )
            SELECT p.image_path, COUNT(e.id) as match_count, MIN(p.embedding <=> e.embedding) as best_distance
            FROM "{table_name}" p
            CROSS JOIN ref_encodings e
            WHERE p.embedding <=> e.embedding < :threshold
            GROUP BY p.image_path
            HAVING COUNT(e.id) >= :min_matches
            ORDER BY match_count DESC, best_distance ASC
        """
        async with self.SessionLocal() as db:
            result = await db.execute(text(query_str), {
                "threshold": threshold,
                "min_matches": min_matches
            })
            rows = result.all()
            return [row[0] for row in rows]

    async def get_closest_matches_debug(self, folder_path: str, encodings: List[List[float]], limit: int = 5) -> List[Dict[str, Any]]:
        table_name = get_event_table_name(folder_path)
        formatted_encodings = [f"'{str(emb)}'" for emb in encodings]
        values_clause = ", ".join([f"({i+1}, {emb}::vector)" for i, emb in enumerate(formatted_encodings)])
        
        query_str = f"""
            WITH ref_encodings(id, embedding) AS (
                VALUES {values_clause}
            )
            SELECT p.image_path, COUNT(e.id) as match_count, MIN(p.embedding <=> e.embedding) as best_distance
            FROM "{table_name}" p
            CROSS JOIN ref_encodings e
            GROUP BY p.image_path
            ORDER BY best_distance ASC
            LIMIT :limit
        """
        async with self.SessionLocal() as db:
            result = await db.execute(text(query_str), {"limit": limit})
            return [{"image_path": row[0], "match_count": row[1], "best_distance": row[2]} for row in result.all()]
