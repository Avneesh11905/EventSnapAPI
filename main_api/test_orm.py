import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import select, func, column, literal
from sqlalchemy.orm import aliased
from sqlalchemy.sql import text, values
from pgvector.sqlalchemy import Vector
from infrastructure.database.models import EventEncodingModel


async def test():
    engine = create_async_engine(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
    )

    encodings = [[0.1] * 512, [0.2] * 512]
    event_code = "TEST"
    threshold = 0.5
    min_matches = 1

    # Format the same way as raw SQL for pgvector cast if needed
    ref_encodings = (
        values(column("id"), column("embedding", Vector(512)), name="ref_encodings")
        .data([(i + 1, str(emb)) for i, emb in enumerate(encodings)])
        .cte("ref_encodings")
    )

    p = aliased(EventEncodingModel, name="p")
    e = ref_encodings

    stmt = (
        select(
            p.image_path,
            func.count(e.c.id).label("match_count"),
            func.min(p.embedding.op("<=>")(e.c.embedding)).label("best_distance"),
        )
        .select_from(p)
        .join(e, literal(True))
        .where(p.event_code == event_code)
        .where(p.embedding.op("<=>")(e.c.embedding) < threshold)
        .group_by(p.image_path)
        .having(func.count(e.c.id) >= min_matches)
        .order_by(text("match_count DESC"), text("best_distance ASC"))
    )

    print(stmt.compile(engine, compile_kwargs={"literal_binds": True}))


asyncio.run(test())
