"""Provider-agnostic text-to-speech interface.

`audio` is a complete, self-describing audio file and `media_type` says
what it is, so the endpoint that serves it and the browser that plays it
never need out-of-band knowledge of a sample rate or an encoding. Both
implementations return WAV, but nothing in the app assumes that — a
provider that only emits MP3 slots in without touching a caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SynthesisResult:
    audio: bytes
    media_type: str
    latency_ms: float
    provider: str
    voice: str
    model: str


class TTSProvider(Protocol):
    name: str
    voice: str
    model: str

    async def synthesize(self, text: str) -> SynthesisResult:
        """Synthesize one chunk of speech.

        Callers pass a sentence at a time rather than a whole reply, and
        that is not an arbitrary chunk size — it is what makes barge-in
        truncation honest. The history can only be rewritten to what the
        user actually heard if the boundaries of "what was played" line
        up with the boundaries of what was synthesized. See
        app/conversation.py's truncate_last_reply.
        """
        ...
