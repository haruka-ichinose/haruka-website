"""Write embedded chunks to Postgres/pgvector — the local counterpart to
``index.py`` (Azure AI Search). Exposes the same ``ensure_schema`` /
``upsert_chunks`` surface so ``build.py`` stays provider-agnostic.

Schema lives in ``sql/*.sql`` (schema-as-code, applied in filename order — add a
new numbered file for each migration); this module applies them and upserts rows
keyed on chunk id.
"""

from __future__ import annotations

from datetime import date
from importlib.resources import files

import asyncpg

from pgvector.asyncpg import register_vector

from .config import settings
from .documents import Chunk

UPSERT_BATCH_SIZE = 500

_SCHEMA_SQL = [
    f.read_text()
    for f in sorted(files("mecha_llm").joinpath("sql").iterdir(), key=lambda f: f.name)
    if f.name.endswith(".sql")
]

# Columns written in order, content_tsv is GENERATED.
_COLUMNS = (
    "id", "doc_id", "chunk_index", "content", "title",
    "source_type", "url", "repo_url", "published_on", "embedding",
)


async def _connect() -> asyncpg.Connection:
    conn = await asyncpg.connect(settings.database_url)
    await register_vector(conn)  # asyncpg send list[float] as vector(384)
    return conn

async def ensure_schema() -> None:
    """Apply the idempotent schema files (extension, table, indexes, migrations)."""
    conn = await _connect()
    try:
        for sql in _SCHEMA_SQL:
            await conn.execute(sql)
    finally:
        await conn.close()

def _row(chunk: Chunk) -> tuple:
    """Chunk -> positional row matching ``_COLUMNS``. Missing optionals are NULL.
    not '' -- ``published_on`` is a real ``date`` column."""
    return (
        chunk.chunk_id,
        chunk.doc_id,
        chunk.chunk_index,
        chunk.text,
        chunk.title,
        chunk.source_type,
        chunk.url,
        chunk.repo_url or None,
        date.fromisoformat(chunk.published_on) if chunk.published_on else None,
        chunk.embedding,
    )

async def upsert_chunks(
    chunks: list[Chunk], *, batch_size: int = UPSERT_BATCH_SIZE
) -> int:
    """Insert or update chunks keyed on id; return the number written."""
    if not chunks:
        return 0

    cols = ", ".join(_COLUMNS)
    placeholders = ", ".join(f"${i}" for i in range(1, len(_COLUMNS) + 1))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _COLUMNS if c != "id")
    sql = (
        f"INSERT INTO chunks ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT (id) DO UPDATE SET {updates}"
    )

    rows = [_row(c) for c in chunks]
    conn = await _connect()
    try:
        async with conn.transaction():
            for i in range(0, len(rows), batch_size):
                await conn.executemany(sql, rows[i : i + batch_size])
    finally:
        await conn.close()
    return len(rows)
