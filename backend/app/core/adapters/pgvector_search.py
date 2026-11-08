"""Thin async wrapper around Postgres/pgvector: one hybrid query.

Hybrid = vector similarity (cosine ``<=>``) fused with Postgres full-text
search (``websearch_to_tsquery`` over the generated ``content_tsv`` column)
via Reciprocal Rank Fusion - the self-hosted stand-in for Azure AI Search's
semantic reranker. Returns the same chunk-dict shape as ``ai_search.hybrid_search``
so ``rag_service`` stays backend-agnostic.

The ``chunks`` table and its HNSW + GIN indexes are created by the offline
pipeline; see ``llm/mecha_llm/sql/001_chunks.sql``.
"""

import logging

import asyncpg
from pgvector.asyncpg import register_vector

from core.config import settings

logger = logging.getLogger(__name__)

# RRF smoothing constant; 60 is the value from the original RRF paper.
RRF_K = 60
# Candidates pulled from each arm before fusion (>= top_k gives fusion depth).
_CANDIDATES = 50
# At most this many chunks per source document, so one long journal entry
# can't fill every slot.
MAX_CHUNKS_PER_DOC = 2

_HYBRID_SQL = """
WITH vector_search AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> $1) AS rank
    FROM chunks
    ORDER BY embedding <=> $1
    LIMIT $3
),
keyword_search AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(content_tsv, query) DESC) AS rank
    FROM chunks, websearch_to_tsquery('english', $2) query
    WHERE content_tsv @@ query
    ORDER BY ts_rank_cd(content_tsv, query) DESC
    LIMIT $3
),
fused AS (
    SELECT
        id,
        COALESCE(1.0 / ($4 + v.rank), 0.0)
        + COALESCE(1.0 / ($4 + k.rank), 0.0) AS score
    FROM vector_search v
    FULL OUTER JOIN keyword_search k USING (id)
)
SELECT c.doc_id, c.content, c.title, c.url, c.repo_url, c.source_type, f.score
FROM fused f
JOIN chunks c USING (id)
ORDER BY f.score DESC
LIMIT $5;
"""


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def create_client() -> asyncpg.Pool:
    """Connection pool for retrieval; register pgvector on every connection."""
    return await asyncpg.create_pool(settings.database_url, init=_init_connection)


async def hybrid_search(
    pool: asyncpg.Pool, *, query_text: str, query_vector: list[float], top_k: int
) -> list[dict]:
    """Return the top-k chunks (content + citation metadata) for a query."""
    # Fetch the whole fused list, then cap per document in score order.
    rows = await pool.fetch(
        _HYBRID_SQL, query_vector, query_text, _CANDIDATES, RRF_K, _CANDIDATES
    )
    per_doc: dict[str, int] = {}
    kept = []
    for r in rows:
        if per_doc.get(r["doc_id"], 0) >= MAX_CHUNKS_PER_DOC:
            continue
        per_doc[r["doc_id"]] = per_doc.get(r["doc_id"], 0) + 1
        kept.append(r)
        if len(kept) == top_k:
            break
    return [
        {
            "content": r["content"],
            "title": r["title"],
            "url": r["url"],
            "repo_url": r["repo_url"] or None,
            "source_type": r["source_type"],
            "score": r["score"],
        }
        for r in kept
    ]
