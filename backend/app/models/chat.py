from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Keep the frontend input's maxLength in sync (chat-widget.tsx). Also keeps the
# Content Safety payload under its 10k-char limit — an oversized request makes
# the call error, and the client fails open.
MAX_MESSAGE_CHARS = 1000


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    # Generous hard cap; the service trims history further (core/guardrails.py).
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    history: list[ChatMessage] = Field(default=[], max_length=50)

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class Citation(BaseModel):
    """A retrieved source the chatbot used, rendered as a clickable link/route."""

    title: str
    source_type: Literal["journal", "project", "experience", "page", "profile", "repo"]
    url: str  # internal route ("/journal/{id}", "/experience") or external URL
    repo_url: str | None = None
