"""Central configuration.

Every tunable in the system is declared here, once.

Two things are deliberately *not* here:

- **The revision and escalation caps** (D-06). They live as module constants in
  ``app/graph/state.py``. A design constraint is not a tuning knob, and putting a
  bound in a settings object is an invitation to raise it.
- **Prompts.** They belong next to the node that sends them, not in a config file
  that hides what each agent actually asks for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_pool(name: str) -> list[str]:
    """Parse a comma-separated pool of API keys from an env var.

    Blank entries (trailing commas, accidental double commas) are dropped rather
    than kept as empty-string "keys" that would fail every call made with them.
    """
    raw = _env(name)
    if not raw:
        return []
    return [key.strip() for key in raw.split(",") if key.strip()]


@dataclass(frozen=True)
class Settings:
    """Immutable settings, read once at import."""

    # --- Providers -------------------------------------------------------
    # Groq is primary, Cerebras is failover (D-01). Each provider is a POOL of
    # keys, not one — ``GROQ_API_KEY=key1,key2,key3`` in .env. The client
    # (Phase 1, app/core/llm.py) rotates within a provider's pool first — moving
    # to the next Groq key on a 429/5xx/connection failure — and only crosses to
    # Cerebras once every Groq key has failed. This stretches free-tier quota
    # across keys before spending the failover tier, without changing which
    # provider is "primary."
    groq_api_keys: list[str] = field(default_factory=lambda: _env_pool("GROQ_API_KEY"))
    cerebras_api_keys: list[str] = field(default_factory=lambda: _env_pool("CEREBRAS_API_KEY"))

    groq_base_url: str = "https://api.groq.com/openai/v1"
    cerebras_base_url: str = "https://api.cerebras.ai/v1"

    # Assumption A, verified in Phase 1 and confirmed from the previous project:
    # the two providers serve the same weights under DIFFERENT model ID strings.
    # A single MODEL constant breaks failover the first time it fires, so the two
    # IDs are separate values from day one and never collapsed into one.
    groq_model: str = "openai/gpt-oss-120b"
    cerebras_model: str = "gpt-oss-120b"

    request_timeout_s: float = 60.0

    # --- Generation defaults ---------------------------------------------
    # Agent decisions want near-determinism; prose synthesis wants a little room.
    decision_temperature: float = 0.0
    synthesis_temperature: float = 0.3
    max_tokens: int = 2048

    # --- Retrieval (Phase 2) ---------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size_words: int = 200
    chunk_overlap_words: int = 40
    retrieval_top_k: int = 8
    # RRF's damping constant. Rank-based fusion, because BM25 and cosine scores
    # live on incomparable scales.
    rrf_k: int = 60

    # --- Paths ------------------------------------------------------------
    root: Path = ROOT
    docs_dir: Path = ROOT / "docs"
    data_dir: Path = ROOT / "data"
    chroma_dir: Path = ROOT / "data" / "chroma_db"
    # BM25 cannot be persisted, so the chunk records are; the keyword index is
    # rebuilt from this file at process start (Phase 2).
    chunks_file: Path = ROOT / "data" / "chunks.json"
    logs_dir: Path = ROOT / "logs"
    trace_file: Path = ROOT / "logs" / "trace.jsonl"
    checkpoint_dir: Path = ROOT / "checkpoints"
    checkpoint_db: Path = ROOT / "checkpoints" / "graph.sqlite"

    # --- Cache (Phase 8) ---------------------------------------------------
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379"))
    cache_similarity_threshold: float = 0.93

    def ensure_dirs(self) -> None:
        """Create the directories the app writes to. Safe to call repeatedly."""
        for path in (self.data_dir, self.logs_dir, self.checkpoint_dir):
            path.mkdir(parents=True, exist_ok=True)

    def missing_keys(self) -> list[str]:
        """Names of API key pools that are entirely empty, for a clear error at startup."""
        missing = []
        if not self.groq_api_keys:
            missing.append("GROQ_API_KEY")
        if not self.cerebras_api_keys:
            missing.append("CEREBRAS_API_KEY")
        return missing


settings = Settings()
