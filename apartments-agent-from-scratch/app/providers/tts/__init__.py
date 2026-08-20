"""TTS provider factory. `TTS_PROVIDER` picks the implementation:
`openai` (default, no extra credentials) or `google` (needs
GCP_TTS_API_KEY). See app/providers/tts/google_tts.py for why a second
implementation exists at all.
"""
from __future__ import annotations

from app.config import TTS_PROVIDER
from app.providers.tts.base import SynthesisResult, TTSProvider  # noqa: F401

_provider: TTSProvider | None = None


def get_tts_provider() -> TTSProvider:
    global _provider
    if _provider is not None:
        return _provider

    if TTS_PROVIDER == "openai":
        from app.providers.tts.openai_tts import OpenAITTSProvider

        _provider = OpenAITTSProvider()
    elif TTS_PROVIDER == "google":
        from app.providers.tts.google_tts import GoogleTTSProvider

        _provider = GoogleTTSProvider()
    else:
        raise ValueError(
            f"Unknown TTS_PROVIDER={TTS_PROVIDER!r}. Use 'openai' or 'google' — "
            "see README 'Swapping a provider' for what another one needs."
        )
    return _provider


def reset_tts_provider_cache() -> None:
    """Drop the memoized provider. Exists for tests, which construct
    providers under patched configuration and must not inherit an
    instance built under a previous one."""
    global _provider
    _provider = None
