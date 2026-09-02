"""Centralized configuration, loaded from environment variables / .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (src/studybot/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    # --- Groq / LLM ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1600

    # --- Course identity ---
    # Shown to the model so it knows which course it's assisting with.
    # Override per deployment with a COURSE_NAME environment variable rather
    # than editing this default — that way the code itself stays identical
    # across every class's branch/Render service, and only deployment config
    # differs. (docs/<course>/index.html's title/greeting text is a separate,
    # unavoidable manual edit per course — a static frontend has no way to
    # read this Python setting at runtime.)
    course_name: str = "Your Course Name"

    # --- Embeddings / retrieval ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 800  # characters, not tokens (kept simple/dependency-free)
    chunk_overlap: int = 150
    top_k: int = 8
    max_history_messages: int = 12  # ~6 turns of conversation

    # --- Image description (for equations/diagrams embedded as pictures,
    # invisible to plain text extraction) ---
    describe_images: bool = True
    vision_model: str = "qwen/qwen3.6-27b"
    image_min_bytes: int = 3000  # skip tiny images (icons, decorative dots)
    image_cache_path: Path = REPO_ROOT / "data" / "image_description_cache.json"
    embedding_cache_path: Path = REPO_ROOT / "data" / "embedding_cache.json"

    # --- Paths ---
    materials_dir: Path = REPO_ROOT / "data" / "materials"
    index_dir: Path = REPO_ROOT / "data" / "vector_index"
    term_aliases_path: Path = REPO_ROOT / "data" / "term_aliases.json"

    # --- Server ---
    allowed_origins: str = "http://localhost:8000"

    # --- Free-tier protection (see cache.py) ---
    max_requests_per_minute: int = 25  # stay a little under Groq's 30/min cap
    cache_ttl_seconds: int = 3600

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
