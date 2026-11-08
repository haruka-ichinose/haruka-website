"""mecha-haruka RAG orchestration: retrieve -> ground -> stream -> cite.

Wraps the turn in OpenTelemetry spans (exported to Azure Monitor when configured;
no-ops otherwise). Safety guardrails (Azure AI Content Safety / Prompt Shields)
are a Phase 7 addition and slot in at the marked hook before retrieval.
"""

import logging
from collections.abc import AsyncIterator

import asyncpg
from openai import AsyncAzureOpenAI
from opentelemetry import trace
from sentence_transformers import SentenceTransformer

from core.adapters import azure_openai, local_embeddings, pgvector_search
from core.adapters.content_safety import ContentSafetyClient
from core.config import settings
from models.chat import ChatRequest, Citation

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("mecha_haruka.rag")

# Keep byte-identical with llm/mecha_llm/eval/rag.py SYSTEM_PROMPT (the eval
# harness can't import the backend).
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

# Citation chips shown under an answer; more than this stops being scannable.
MAX_CITATIONS = 4

# Shown when Content Safety blocks the input (prompt injection or harmful content).
REFUSAL = (
    "I can't help with that. I'm Mecha Haruka — I can tell you about Haruka's "
    "projects, work and education, skills and tech stack, career, public GitHub "
    "repositories, dev journal, and how this site and chatbot were built. "
    "Ask me about any of those and I'm happy to help."
)


class RagService:
    def __init__(
        self,
        openai_client: AsyncAzureOpenAI,
        search_pool: asyncpg.Pool,
        embed_client: SentenceTransformer,
        safety_client: ContentSafetyClient | None = None,
    ) -> None:
        self._openai = openai_client
        self._search = search_pool
        self._embed = embed_client
        self._safety = safety_client

    async def stream_answer(self, req: ChatRequest) -> AsyncIterator[dict]:
        """Yield event dicts: {type: token|citations|done, ...}."""
        with tracer.start_as_current_span("rag.turn") as span:
            span.set_attribute("rag.message_length", len(req.message))
            span.set_attribute("rag.history_length", len(req.history))

            # Safety gate: Prompt Shields + harmful-content check on the input.
            # Output is screened by Azure OpenAI's default completion content filter,
            # so we keep streaming rather than buffering to moderate the response.
            if self._safety is not None:
                with tracer.start_as_current_span("rag.safety") as safety_span:
                    blocked = await self._safety.detect_prompt_attack(req.message)
                    if not blocked:
                        blocked = await self._safety.is_text_harmful(req.message)
                    safety_span.set_attribute("rag.input_blocked", blocked)
                if blocked:
                    span.set_attribute("rag.blocked", True)
                    logger.warning("Input blocked by Content Safety")
                    yield {"type": "token", "delta": REFUSAL}
                    yield {"type": "citations", "citations": []}
                    yield {"type": "done"}
                    return

            with tracer.start_as_current_span("rag.retrieve") as retrieve_span:
                query_vector = await local_embeddings.embed_query(self._embed, req.message)
                chunks = await pgvector_search.hybrid_search(
                    self._search,
                    query_text=req.message,
                    query_vector=query_vector,
                    top_k=settings.rag_top_k,
                )
                retrieve_span.set_attribute("rag.retrieved_count", len(chunks))

            messages = self._build_messages(req, chunks)

            with tracer.start_as_current_span("rag.generate"):
                async for delta in azure_openai.stream_chat(self._openai, messages):
                    yield {"type": "token", "delta": delta}

            citations = self._citations(chunks)
            span.set_attribute("rag.citation_count", len(citations))
            yield {"type": "citations", "citations": [c.model_dump() for c in citations]}
            yield {"type": "done"}

    def _build_messages(self, req: ChatRequest, chunks: list[dict]) -> list[dict]:
        if chunks:
            context = "\n\n".join(
                f"[{i + 1}] {c['title']} ({c['url']})\n{c['content']}"
                for i, c in enumerate(chunks)
            )
        else:
            context = "(no relevant context found)"

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": m.role, "content": m.content} for m in req.history)
        messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": req.message})
        return messages

    def _citations(self, chunks: list[dict]) -> list[Citation]:
        seen: set[str] = set()
        citations: list[Citation] = []
        for c in chunks:
            if c["url"] in seen:
                continue
            seen.add(c["url"])
            if len(citations) == MAX_CITATIONS:
                break
            citations.append(
                Citation(
                    title=c["title"],
                    source_type=c["source_type"],
                    url=c["url"],
                    repo_url=c["repo_url"],
                )
            )
        return citations
