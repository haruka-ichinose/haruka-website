"""Orchestrate the offline pipeline: ingest -> chunk -> embed -> upsert.

Run as a module:

    uv run python -m mecha_llm.build              # full build into Azure AI Search
    uv run python -m mecha_llm.build --dry-run    # ingest + chunk only, no Azure calls
    uv run python -m mecha_llm.build --skip-journals   # skip the private GitHub source
    uv run python -m mecha_llm.build --skip-repos      # skip the public repo READMEs

The journal and repo sources need ``GITHUB_TOKEN``; the embed/index stages need the Azure
OpenAI and AI Search settings. ``--dry-run`` exercises everything that runs without
Azure, which is handy for sanity-checking chunk counts.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from .chunking import chunk_documents
from .config import settings
from .documents import SourceDoc
from .embeddings_provider import embed_chunks
from .pg_store import ensure_schema, upsert_chunks
from .sources.content_json import load_content_docs
from .sources.github_repos import fetch_repo_docs
from .sources.journals import fetch_journal_docs
from .sources.profile import load_profile_docs
from .sources.routes import route_docs

logger = logging.getLogger("mecha_llm.build")


async def gather_source_docs(
    *, skip_journals: bool = False, skip_repos: bool = False
) -> list[SourceDoc]:
    """Collect documents from every content source."""
    docs: list[SourceDoc] = []

    if skip_journals:
        logger.info("Skipping journal source (--skip-journals).")
    else:
        journals = await fetch_journal_docs()
        logger.info("Fetched %d journal entries.", len(journals))
        docs.extend(journals)

    content = load_content_docs(settings.content_json_path)
    logger.info("Loaded %d project/experience docs from content.json.", len(content))
    docs.extend(content)

    routes = route_docs()
    logger.info("Built %d navigation docs.", len(routes))
    docs.extend(routes)

    profile = load_profile_docs(settings.profile_md_path)
    logger.info("Loaded %d profile sections.", len(profile))
    docs.extend(profile)

    if skip_repos:
        logger.info("Skipping repo source (--skip-repos).")
    else:
        repos = await fetch_repo_docs()
        logger.info("Fetched %d repo READMEs.", len(repos))
        docs.extend(repos)

    return docs


async def build(
    *, dry_run: bool = False, skip_journals: bool = False, skip_repos: bool = False
) -> None:
    docs = await gather_source_docs(skip_journals=skip_journals, skip_repos=skip_repos)
    chunks = chunk_documents(docs)
    logger.info("Chunked %d documents into %d chunks.", len(docs), len(chunks))

    if dry_run:
        logger.info("Dry run: skipping embed + index. %d chunks ready.", len(chunks))
        return

    if not chunks:
        logger.warning("No chunks to index; nothing to do.")
        return

    embedded = await embed_chunks(chunks)
    logger.info("Embedded %d chunks (%d dims).", len(embedded), settings.embedding_dimensions)

    await ensure_schema()
    logger.info("Ensured pgvector schema is up to date.")

    uploaded = await upsert_chunks(embedded)
    logger.info("Upserted %d/%d chunks into Postgres/pgvector.", uploaded, len(embedded))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the mecha-haruka RAG index.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ingest and chunk only; skip embedding and indexing (no Azure calls).",
    )
    parser.add_argument(
        "--skip-journals",
        action="store_true",
        help="Skip the private GitHub journal source (no GITHUB_TOKEN needed).",
    )
    parser.add_argument(
        "--skip-repos",
        action="store_true",
        help="Skip the public GitHub repo README source (no GITHUB_TOKEN needed).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(
        build(dry_run=args.dry_run, skip_journals=args.skip_journals, skip_repos=args.skip_repos)
    )


if __name__ == "__main__":
    main()
