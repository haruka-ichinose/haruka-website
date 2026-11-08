from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the offline pipeline and eval harness.

    All fields default to empty/placeholder values so the offline, Azure-free
    stages (ingestion, chunking) and the unit tests can import and run without a
    populated ``.env``. The embed/index/eval stages validate the values they
    need at call time (see ``require``).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Content sources ---
    github_token: str = ""
    github_repo: str = "Jade1724/obsidian"
    journals_base_path: str = "Dev/Journals"
    content_json_path: str = "../frontend/lib/content.json"
    profile_md_path: str = "content/profile.md"
    # Public repos to index beyond the github links in content.json ("owner/name").
    extra_repos: list[str] = []

    # --- Azure AI Foundry ---
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    # Used for both generation (the system under test) and grading (the RAGAS /
    # DeepEval judge). Mirror the backend's deployment so the eval harness
    # measures what production actually runs.
    azure_openai_chat_deployment: str = "DeepSeek-V4-Flash"
    # Not every Foundry model accepts `temperature` (reasoning models reject it).
    # Set to null/empty to omit the parameter from requests entirely — including
    # the judges', which otherwise pin it to 0 for deterministic grading.
    chat_temperature: float | None = 0.3
    embedding_dimensions: int = 384

    # --- Local embeddings (sentence-transformers) ---
    embedding_provider: str = "local"   # "local" | "azure"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # --- Azure AI Search ---
    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index: str = "mecha-haruka"

    # --- Chunking ---
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 60

    # --- Observability (OpenTelemetry -> Azure Monitor / Application Insights) ---
    applicationinsights_connection_string: str = ""
    otel_service_name: str = "mecha-haruka"

    # --- Self-hosting pgvector config ---
    database_url: str = ""

    def require(self, *fields: str) -> None:
        """Raise if any named setting is empty — call before a stage that needs it."""
        missing = [f for f in fields if not getattr(self, f)]
        if missing:
            raise RuntimeError(
                "Missing required settings: "
                + ", ".join(sorted(missing))
                + ". Set them in llm/.env (see .env.example)."
            )


settings = Settings()
