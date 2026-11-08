"""Thin async wrapper around Azure AI Foundry: streaming chat generation.

Mirrors the simplicity of ``email.py`` — module-level functions over a client
that ``main.py`` creates once (in the lifespan) and shares via ``app.state``.
Query embeddings moved to ``local_embeddings`` (bge, on-device); this module now
only handles chat generation.

The model is chosen entirely by ``settings.azure_openai_chat_deployment``, so
this is the single place a model swap touches. Request parameters are kept
minimal and configurable because Foundry hosts non-OpenAI models (DeepSeek et al)
that don't all accept the same knobs.
"""

import logging
from collections.abc import AsyncIterator

from openai import AsyncAzureOpenAI

from core.config import settings

logger = logging.getLogger(__name__)


def create_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


async def stream_chat(
    client: AsyncAzureOpenAI, messages: list[dict]
) -> AsyncIterator[str]:
    """Yield content deltas from a streamed chat completion.

    Only ``delta.content`` is yielded. Reasoning models stream their thinking in
    a separate ``reasoning_content`` field, which is deliberately dropped here —
    it must never reach the user — at the cost of a slower first visible token.
    """
    params: dict = {}
    if settings.chat_temperature is not None:
        params["temperature"] = settings.chat_temperature
    if settings.chat_max_tokens is not None:
        params["max_tokens"] = settings.chat_max_tokens

    stream = await client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=messages,
        stream=True,
        **params,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
