from typing import Literal

from pydantic import BaseModel, field_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []

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
