"""STT provider factory — the same selector pattern as
app/providers/llm/__init__.py. `STT_PROVIDER` in the environment picks
the implementation; nothing else in the app names a vendor.

Unlike the LLM factory there is no mock default, and that is intentional.
The LLM has a mock because the app must run end to end with zero
credentials — that is a hard requirement, and a scripted stand-in
satisfies it. Voice has no such requirement: it is a bonus mode, not
the deliverable, and a fake transcriber that returns canned text would
demonstrate nothing while looking in a screenshot exactly like a working
one. Without a key, voice mode is reported as unavailable, with the
reason, and the text interface is untouched.
"""
from __future__ import annotations

from app.config import STT_PROVIDER
from app.providers.stt.base import STTProvider, TranscriptResult  # noqa: F401

_provider: STTProvider | None = None


def get_stt_provider() -> STTProvider:
    global _provider
    if _provider is not None:
        return _provider

    if STT_PROVIDER == "openai":
        from app.providers.stt.openai_stt import OpenAISTTProvider

        _provider = OpenAISTTProvider()
    else:
        raise ValueError(
            f"Unknown STT_PROVIDER={STT_PROVIDER!r}. Only 'openai' is implemented — "
            "see README 'Swapping a provider' for what another one needs."
        )
    return _provider


def reset_stt_provider_cache() -> None:
    """Drop the memoized provider. Exists for tests, which construct
    providers under patched configuration and must not inherit an
    instance built under a previous one."""
    global _provider
    _provider = None
