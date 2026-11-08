"""Shared data structures for the pipeline.

A ``SourceDoc`` is one logical document from a content source (a journal entry, a
project, an experience, a navigable page, a profile section, or a repo README). The
chunker turns each ``SourceDoc`` into one or more ``Chunk``s, which are what gets
embedded and indexed. Both carry the metadata needed for the chatbot to cite a
clickable link or route.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Allowed values for ``source_type`` — kept in sync with the citation rendering
# on the frontend (internal route vs. external link).
SOURCE_TYPES = ("journal", "project", "experience", "page", "profile", "repo")


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    title: str
    text: str
    source_type: str
    url: str  # internal route ("/journal/{id}", "/experience") or external URL
    repo_url: str | None = None
    published_on: str | None = None  # ISO date string, when known

    def __post_init__(self) -> None:
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(
                f"Unknown source_type {self.source_type!r}; expected one of {SOURCE_TYPES}"
            )


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_id}_{chunk_index}" (Azure Search keys allow only [A-Za-z0-9_\-=])
    doc_id: str
    chunk_index: int
    text: str
    title: str
    source_type: str
    url: str
    repo_url: str | None = None
    published_on: str | None = None
    embedding: list[float] | None = field(default=None, compare=False)

    def to_search_document(self) -> dict:
        """Serialize to an Azure AI Search index document (see ``index.py`` schema)."""
        doc = {
            "id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "content": self.text,
            "title": self.title,
            "source_type": self.source_type,
            "url": self.url,
            "repo_url": self.repo_url or "",
            "published_on": self.published_on or "",
        }
        if self.embedding is not None:
            doc["embedding"] = self.embedding
        return doc
