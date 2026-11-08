"""System-under-test for evaluation: retrieve -> ground -> generate.

The ``llm/`` package can't import the backend, so this mirrors the backend's
RAG path (``backend/app/services/rag_service.py``) and returns both the answer
and the retrieved contexts that RAGAS needs. Non-streaming (eval wants the full
answer). Keep ``SYSTEM_PROMPT``, the context format, and the retrieval constants
in sync with the backend's.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import asyncpg
from openai import AsyncAzureOpenAI
from pgvector.asyncpg import register_vector

from ..config import settings
from ..embeddings_local import embed_texts

# RRF fusion constants; mirror backend/app/core/adapters/pgvector_search.py.
RRF_K = 60
CANDIDATES = 50
DEFAULT_TOP_K = 8
MAX_CHUNKS_PER_DOC = 2

# Keep byte-identical with backend/app/services/rag_service.py SYSTEM_PROMPT.
SYSTEM_PROMPT = """\
You are Mecha Haruka, an AI persona of Haruka — a software/AI engineer. You speak \
on Haruka's portfolio website to help visitors learn about Haruka. You know about \
Haruka's projects, work and education history, skills and tech stack, background \
and career direction, public GitHub repositories, dev journal, and this website.

Rules:
- Ground every answer in the provided context. Never invent projects, employers, \
dates, repositories, contact details, or links. If the context only partly covers \
the question, answer the part it supports and say what you don't know.
- Never dead-end. If you can't answer, don't just say you don't have the \
information: say what the context does cover that is closest, suggest 2–3 \
specific questions the visitor could ask instead, and link the most relevant page.
- Write site pages as markdown links: [Projects](/), [Experience](/experience), \
[dev journal](/journal), or a journal entry route from the context. Only link \
routes and URLs that appear in the context or this list.
- Professional and light personal questions about Haruka are in scope. Decline \
only harmful or abusive requests, requests clearly unrelated to Haruka or this \
site, and personal details the context doesn't contain.
- You are an AI persona of Haruka, not Haruka themselves — don't claim to be a \
person.
- Be concise, warm, and professional: 1–3 short paragraphs. When your answer is \
thin, end with a suggested follow-up question.
- The UI shows the sources you used as links, so you don't need to paste raw \
source URLs into your prose."""


@dataclass(frozen=True)
class RagResult:
    question: str
    answer: str
    contexts: list[str]
    urls: list[str]  # citation urls + repo urls of the contexts, for the route hit rate


# Hybrid retrieval mirroring the backend's pgvector_search adapter: vector
# (cosine) + Postgres FTS fused with Reciprocal Rank Fusion.
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
SELECT c.doc_id, c.content, c.title, c.url, c.repo_url
FROM fused f
JOIN chunks c USING (id)
ORDER BY f.score DESC
LIMIT $5;
"""


async def _retrieve(question: str, vector: list[float], top_k: int) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(settings.database_url)
    await register_vector(conn)
    try:
        rows = await conn.fetch(
            _HYBRID_SQL, vector, question, CANDIDATES, RRF_K, CANDIDATES
        )
    finally:
        await conn.close()
    # Per-document cap in score order, as in the backend adapter.
    per_doc: dict[str, int] = {}
    kept = []
    for r in rows:
        if per_doc.get(r["doc_id"], 0) >= MAX_CHUNKS_PER_DOC:
            continue
        per_doc[r["doc_id"]] = per_doc.get(r["doc_id"], 0) + 1
        kept.append(r)
        if len(kept) == top_k:
            break
    return kept


async def _generate(question: str, chunks: list[asyncpg.Record]) -> str:
    # Same "[n] title (url)" header as RagService._build_messages.
    block = "\n\n".join(
        f"[{i + 1}] {c['title']} ({c['url']})\n{c['content']}"
        for i, c in enumerate(chunks)
    ) or "(no relevant context found)"
    client = AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
    # Keep in lockstep with backend/app/core/adapters/azure_openai.py — this is
    # the same generation call, non-streamed, so the eval measures production.
    params: dict = {}
    if settings.chat_temperature is not None:
        params["temperature"] = settings.chat_temperature

    async with client:
        resp = await client.chat.completions.create(
            model=settings.azure_openai_chat_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"Context:\n{block}"},
                {"role": "user", "content": question},
            ],
            **params,
        )
    return (resp.choices[0].message.content or "").strip()


async def _answer_with_vector(
    question: str, vector: list[float], top_k: int
) -> RagResult:
    chunks = await _retrieve(question, vector, top_k)
    text = await _generate(question, chunks)
    return RagResult(
        question=question,
        answer=text,
        contexts=[c["content"] for c in chunks],
        urls=[u for c in chunks for u in (c["url"], c["repo_url"]) if u],
    )


async def answer(question: str, *, top_k: int = DEFAULT_TOP_K) -> RagResult:
    settings.require(
        "azure_openai_endpoint",
        "azure_openai_api_key",
        "database_url",
    )
    vector = (await embed_texts([question], is_query=True))[0]
    return await _answer_with_vector(question, vector, top_k)


async def answer_all(
    questions: list[str], *, top_k: int = DEFAULT_TOP_K, concurrency: int = 4
) -> list[RagResult]:
    settings.require(
        "azure_openai_endpoint",
        "azure_openai_api_key",
        "database_url",
    )
    # Embed all queries in one batch up front: loads the model once and never
    # runs encode() from multiple worker threads (concurrent mps inference can
    # crash natively). Retrieval + generation below stay concurrent (I/O bound).
    vectors = await embed_texts(questions, is_query=True)
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str, vec: list[float]) -> RagResult:
        async with sem:
            return await _answer_with_vector(q, vec, top_k)

    return await asyncio.gather(
        *(one(q, v) for q, v in zip(questions, vectors))
    )
