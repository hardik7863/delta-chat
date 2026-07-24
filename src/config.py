"""Central configuration. Config-over-hardcoding: every threshold, model, and path
lives here and is overridable via environment variables (prefix DELTA_) or a .env file.

Nothing in the structural pipeline should hardcode a value that a reviewer might
reasonably want to tune. LLM/model settings are isolated so the provider is swappable.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DELTA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- paths ----
    repo_root: Path = REPO_ROOT
    runs_dir: Path = REPO_ROOT / "runs"
    samples_dir: Path = REPO_ROOT / "data" / "samples"

    # ---- LLM (isolated so provider is swappable) ----
    llm_provider: str = "anthropic"  # anthropic | openai | ollama | echo(test)
    llm_model: str = "claude-opus-4-8"          # hard reasoning / judge
    llm_model_fast: str = "claude-haiku-4-5"    # routine chat answers
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_max_tokens: int = 1024
    # NOTE: Opus 4.8 / Haiku 4.5 reject temperature/top_p (400). We never send sampling
    # params to Anthropic; chat determinism comes from grounding + the deterministic
    # structural delta, not from temperature. Kept only for providers that accept it.
    llm_temperature: float = 0.0

    # ---- ingestion ----
    ocr_dpi: int = 300
    ocr_engine: str = "tesseract"  # tesseract | paddle

    # ---- delta alignment thresholds ----
    tag_fuzzy_threshold: float = 88.0   # rapidfuzz 0-100; below this, not the same tag
    text_fuzzy_threshold: float = 82.0  # untagged text similarity to consider a match
    spatial_move_threshold: float = 12.0  # pts; centroid shift above this = "moved"
    spatial_match_max_dist: float = 60.0  # pts; max centroid distance for spatial match
    bbox_round: int = 1  # rounding for stable element id hashing

    # ---- retrieval ----
    retrieval_top_k: int = 8
    retrieval_min_score: float = 0.12   # below this, chat refuses instead of answering
    embedding_backend: str = "hash"     # hash(deterministic, zero-dep) | anthropic | bge

    def ensure_dirs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
