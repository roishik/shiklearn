"""Provider-agnostic speech-to-text interface.

Same shape as app/providers/llm/base.py, for the same reason: nothing
outside app/providers/stt/* may import a vendor endpoint or SDK, so
"swap the transcription vendor" stays a one-line config change plus one
new file.

The interface takes an ENCODED AUDIO FILE, not raw PCM. That is a
deliberate difference from a server-side-VAD design, where the server
holds a running PCM buffer and decides where the utterance ends. Here
the browser does endpointing and hands over one finished utterance, so
the natural unit at this boundary is a file, and the provider does not
need to know a sample rate the container already states.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TranscriptResult:
    text: str
    latency_ms: float
    provider: str
    model: str


class STTProvider(Protocol):
    name: str
    model: str

    async def transcribe(self, audio: bytes, *, filename: str, content_type: str) -> TranscriptResult:
        """Transcribe one complete utterance.

        Batch per utterance, not streaming. Streaming ASR would let the
        first words be transcribed while the user is still speaking, and
        that is genuinely what a production voice stack does — see
        DESIGN_DOC.md, where this is recorded as a known divergence with
        a named upgrade path rather than hidden.
        """
        ...
