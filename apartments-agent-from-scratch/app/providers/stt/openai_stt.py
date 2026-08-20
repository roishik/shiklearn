"""OpenAI speech-to-text, via the REST transcriptions endpoint.

Uses `httpx` directly rather than the `openai` SDK, matching every other
provider in this repo: it keeps the dependency list at four packages, and
it keeps "swap this one file to change vendors" literally true, because
no SDK-specific type ever escapes this module.

Model: `gpt-4o-mini-transcribe`, with `whisper-1` as a fallback. The
fallback is not defensive padding — a transcription model can be absent
from a given account tier, and the failure mode without it is that voice
mode simply does not work for that reviewer, with an opaque 400 as the
only clue. Falling back to the most universally available model on the
API turns that into a slower answer instead of no answer.

Why transcription accuracy matters more here than in a generic voice
app: nearly every user turn in this domain contains a locality or
neighborhood name, often a Hebrew place name transliterated into
English. "Ra'anana" heard as "Ranana" or "Kfar Saba" split into two
unrelated words is not a cosmetic error — it is the difference between a
resolved entity and a failed one. That is also why the entity resolver's
fuzzy matching earns its keep on the voice path specifically: it is the
layer that absorbs what the transcriber gets slightly wrong.
"""
from __future__ import annotations

import logging
import time

import httpx

from app.config import OPENAI_API_KEY, OPENAI_STT_FALLBACK_MODEL, OPENAI_STT_MODEL
from app.providers.stt.base import TranscriptResult

logger = logging.getLogger("stt")

_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"

# A short vocabulary hint. The transcription API accepts a `prompt` as a
# style/vocabulary nudge, and this domain's failure mode is proper nouns,
# so spending a handful of tokens on the shape of the vocabulary is worth
# more than it costs. Deliberately NOT a list of specific localities: this
# is a hint, and biasing it toward a fixed set would make it likelier to
# hear one of those names when the user said a different one.
_VOCAB_HINT = (
    "A question about buying a home in Israel — localities, neighborhoods, home "
    "prices, mortgages, or value for money. May contain transliterated Hebrew "
    "place names such as Ra'anana, Kfar Saba, or Be'er Sheva."
)


class OpenAISTTProvider:
    name = "openai"

    def __init__(
        self,
        model: str = OPENAI_STT_MODEL,
        fallback_model: str = OPENAI_STT_FALLBACK_MODEL,
    ):
        self.model = model
        self.fallback_model = fallback_model
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY (or lowercase openai_api_key) is not set — "
                "cannot construct OpenAISTTProvider."
            )

    async def _call(self, model: str, audio: bytes, filename: str, content_type: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                _TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files={"file": (filename, audio, content_type)},
                data={"model": model, "response_format": "json", "prompt": _VOCAB_HINT},
            )
        resp.raise_for_status()
        return resp.json().get("text", "")

    async def transcribe(self, audio: bytes, *, filename: str, content_type: str) -> TranscriptResult:
        started = time.perf_counter()
        model_used = self.model
        try:
            text = await self._call(self.model, audio, filename, content_type)
        except httpx.HTTPStatusError as exc:
            # Only for "this account cannot use that model". Catching every
            # HTTP error here re-uploaded the whole utterance on a 401 or a
            # 429 — doubling latency and cost exactly when the API is
            # already refusing, and hiding the reason, since the only trace
            # was the model name in the response. A 401 will fail the same
            # way twice; a 429 wants backoff, not an immediate retry.
            if exc.response.status_code not in (400, 403, 404):
                raise
            logger.info(
                "STT model %s unavailable (HTTP %s); retrying with %s",
                self.model,
                exc.response.status_code,
                self.fallback_model,
            )
            model_used = self.fallback_model
            text = await self._call(self.fallback_model, audio, filename, content_type)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return TranscriptResult(
            text=text.strip(), latency_ms=latency_ms, provider=self.name, model=model_used
        )
