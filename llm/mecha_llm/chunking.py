"""Markdown-aware, token-bounded chunking.

Splits each document on markdown headings, then packs the prose under each heading
into token-bounded windows with overlap. The heading is prepended to every chunk it
covers so each chunk carries its local context into the embedding. Token counting is
pluggable (``count_tokens``) so tests can run offline without tiktoken.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

from .config import settings
from .documents import Chunk, SourceDoc

CountTokens = Callable[[str], int]

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def _tiktoken_counter() -> CountTokens:
    """Default counter using tiktoken's o200k_base (gpt-4o / text-embedding-3)."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    return lambda text: len(enc.encode(text))


def default_count_tokens(text: str) -> int:
    try:
        return _tiktoken_counter()(text)
    except Exception:
        # Offline fallback: rough word-based estimate.
        return len(text.split())


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs. Preamble has an empty heading.

    Lines inside fenced code blocks are never headings (``# comment`` in a shell
    snippet would otherwise split the document)."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            sections.append((m.group(1), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(lines).strip()) for h, lines in sections if "\n".join(lines).strip() or h]


def _paragraphs(body: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def _split_oversized(segment: str, max_tokens: int, count: CountTokens) -> list[str]:
    """Split a too-large segment by sentences, then by words as a last resort."""
    if count(segment) <= max_tokens:
        return [segment]

    out: list[str] = []
    buf: list[str] = []
    for sentence in _SENTENCE_RE.split(segment):
        if not sentence:
            continue
        if count(sentence) > max_tokens:  # a single sentence is still too big
            if buf:
                out.append(" ".join(buf))
                buf = []
            out.extend(_split_words(sentence, max_tokens, count))
            continue
        candidate = " ".join([*buf, sentence])
        if buf and count(candidate) > max_tokens:
            out.append(" ".join(buf))
            buf = [sentence]
        else:
            buf.append(sentence)
    if buf:
        out.append(" ".join(buf))
    return out


def _split_words(text: str, max_tokens: int, count: CountTokens) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for word in text.split():
        candidate = " ".join([*buf, word])
        if buf and count(candidate) > max_tokens:
            out.append(" ".join(buf))
            buf = [word]
        else:
            buf.append(word)
    if buf:
        out.append(" ".join(buf))
    return out


def _pack(segments: list[str], max_tokens: int, overlap_tokens: int, count: CountTokens) -> list[str]:
    """Greedily pack segments into windows under max_tokens, with token overlap."""
    chunks: list[str] = []
    current: list[str] = []

    def emit() -> list[str]:
        chunks.append("\n\n".join(current))
        # Carry trailing segments as overlap for the next window.
        tail: list[str] = []
        total = 0
        for seg in reversed(current):
            t = count(seg)
            if total + t > overlap_tokens:
                break
            tail.insert(0, seg)
            total += t
        return tail

    for segment in segments:
        for piece in _split_oversized(segment, max_tokens, count):
            tentative = [*current, piece]
            if current and count("\n\n".join(tentative)) > max_tokens:
                current = emit()
            current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(
    doc: SourceDoc,
    *,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    count_tokens: CountTokens = default_count_tokens,
) -> list[Chunk]:
    max_tokens = max_tokens if max_tokens is not None else settings.chunk_max_tokens
    overlap_tokens = overlap_tokens if overlap_tokens is not None else settings.chunk_overlap_tokens

    chunks: list[Chunk] = []
    index = 0
    for heading, body in _split_sections(doc.text):
        prefix = heading.strip()
        budget = max(1, max_tokens - (count_tokens(prefix) if prefix else 0))
        for piece in _pack(_paragraphs(body), budget, overlap_tokens, count_tokens):
            text = f"{prefix}\n\n{piece}" if prefix else piece
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_{index}",
                    doc_id=doc.doc_id,
                    chunk_index=index,
                    text=text,
                    title=doc.title,
                    source_type=doc.source_type,
                    url=doc.url,
                    repo_url=doc.repo_url,
                    published_on=doc.published_on,
                )
            )
            index += 1
    return chunks


def chunk_documents(
    docs: list[SourceDoc],
    *,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    count_tokens: CountTokens = default_count_tokens,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(
            chunk_document(
                doc,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                count_tokens=count_tokens,
            )
        )
    return out
