from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import DEFAULT_MODELS, ProviderConfig, normalize_provider

DEFAULT_COMPACT_THRESHOLD_TOKENS = 500
DEFAULT_COMPACT_KEEP_MESSAGES = 4

# provider -> (api key env var, base url env var)
_PROVIDER_ENV = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "custom": ("CUSTOM_API_KEY", "CUSTOM_BASE_URL"),
    "gemini": ("GEMINI_API_KEY", None),
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "ollama": (None, "OLLAMA_BASE_URL"),
    "openrouter": ("OPENROUTER_API_KEY", None),
}


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Paths: repo root, dataset directory, state directory (holds `User.md` files).
    - Compact memory: token threshold that triggers compaction, and how many
      recent messages survive a compaction verbatim.
    - Providers: main chat model and judge model.
    - `live`: when False (default) both agents use the deterministic offline path,
      so the benchmark and tests run without API keys.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    live: bool = False
    profile_min_confidence: float = 0.5


def _load_dotenv(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(root / ".env", override=False)


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Read `<prefix>_PROVIDER`, `<prefix>_MODEL`, `<prefix>_TEMPERATURE` (+ provider key/URL)."""

    raw_provider = os.getenv(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai")
    provider = normalize_provider(raw_provider)
    model_name = os.getenv(f"{prefix}_MODEL") or (
        fallback.model_name if fallback and fallback.provider == provider else DEFAULT_MODELS[provider]
    )
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0"))

    key_env, url_env = _PROVIDER_ENV[provider]
    api_key = os.getenv(key_env) if key_env else None
    if provider == "gemini" and not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv(url_env) if url_env else None

    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables (optionally from `.env`) and return a LabConfig.

    Env knobs:
    - LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE
    - JUDGE_PROVIDER / JUDGE_MODEL (default: same as LLM_*)
    - OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY,
      OLLAMA_BASE_URL, CUSTOM_BASE_URL / CUSTOM_API_KEY
    - COMPACT_THRESHOLD_TOKENS / COMPACT_KEEP_MESSAGES
    - LAB_MODE=live to call real models (default: offline)
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS)),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES)),
        model=model,
        judge_model=judge_model,
        live=os.getenv("LAB_MODE", "offline").strip().lower() == "live",
        profile_min_confidence=float(os.getenv("PROFILE_MIN_CONFIDENCE", "0.5")),
    )
