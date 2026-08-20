"""OpenAI text-to-speech, via the REST speech endpoint.

The default, and the reason is credentials rather than quality: the app
already needs an OpenAI key to run against a real model, so choosing
OpenAI for speech means voice mode costs a reviewer nothing extra to try.
A second vendor for the voice half would have made the bonus feature the
hardest part of the setup, which is backwards.

`response_format="wav"` rather than mp3. WAV is larger on the wire, which
is irrelevant over localhost, and in exchange the browser decodes it with
no codec dependency and the server never transcodes anything. The one
place format would matter — a remote deployment paying for bandwidth —
is a one-line change here and nowhere else.

Text arrives with markdown already stripped by the caller: the model
writes `**Ra'anana**` and tables, and a synthesizer reads those literally.
See static/markdown.js's stripMarkdown.
"""
from __future__ import annotations

import time

import httpx

from app.config import OPENAI_API_KEY, OPENAI_TTS_MODEL, OPENAI_TTS_VOICE
from app.providers.tts.base import SynthesisResult

_SPEECH_URL = "https://api.openai.com/v1/audio/speech"

# gpt-4o-mini-tts takes a free-text style instruction. Worth setting
# explicitly: the default read of a paragraph full of decimals and
# place names is a fast, flat one, and this agent's whole value is
# that a number came from somewhere. Slower and more even makes "zero
# point seven one" land as a figure rather than filler.
_VOICE_INSTRUCTIONS = (
    "Speak as a real-estate analyst briefing a client: calm, measured, and "
    "unhurried. Read numbers, prices, and locality/neighborhood names clearly "
    "and slightly slower than the surrounding prose. Do not sound salesy or "
    "enthusiastic."
)


class OpenAITTSProvider:
    name = "openai"

    def __init__(self, voice: str = OPENAI_TTS_VOICE, model: str = OPENAI_TTS_MODEL):
        self.voice = voice
        self.model = model
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY (or lowercase openai_api_key) is not set — "
                "cannot construct OpenAITTSProvider."
            )

    async def synthesize(self, text: str) -> SynthesisResult:
        started = time.perf_counter()
        payload: dict[str, object] = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": "wav",
        }
        # Only the newer speech models accept `instructions`; sending it to
        # tts-1 is a 400. Gate on the model rather than catching the error,
        # so a misconfiguration fails loudly instead of silently retrying.
        if self.model.startswith("gpt-4o"):
            payload["instructions"] = _VOICE_INSTRUCTIONS

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                _SPEECH_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=payload,
            )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return SynthesisResult(
            audio=resp.content,
            media_type="audio/wav",
            latency_ms=latency_ms,
            provider=self.name,
            voice=self.voice,
            model=self.model,
        )
