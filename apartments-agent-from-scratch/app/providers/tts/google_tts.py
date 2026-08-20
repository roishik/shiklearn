"""Google Cloud Text-to-Speech — the second TTS implementation.

This file exists to make the provider abstraction a demonstrated property
rather than a claim. An interface with exactly one implementation has
never been tested as an interface; it is a class with extra steps. Two
real vendors behind it, switchable with `TTS_PROVIDER=google`, is what
makes "we could move to ElevenLabs or Cartesia" a statement about work
already proven possible rather than an intention.

Authenticated with a plain API key against the REST `text:synthesize`
endpoint, deliberately not the `google-cloud-texttospeech` SDK — that SDK
expects service-account credentials (a JSON keyfile plus ADC), which is a
heavier setup than an API key for a demo, and it would add a dependency to
a four-package project.

Voice tier: Neural2, explicitly not Studio or Chirp3-HD. Neural2 is a
clear step up from Standard in prosody without the premium tiers' pricing.

SECURITY: the key goes in the `X-Goog-Api-Key` HEADER, never as a `?key=`
query parameter. This is a real bug caught once and worth not repeating —
httpx logs full request URLs at INFO level, so a query-string key writes
the live credential into the server log in plaintext on every call.
Headers are not in that log line. app/config.py also pins httpx's logger
to WARNING as a second line of defense.
"""
from __future__ import annotations

import base64
import time

import httpx

from app.config import GCP_TTS_API_KEY, GOOGLE_TTS_LANGUAGE_CODE, GOOGLE_TTS_VOICE
from app.providers.tts.base import SynthesisResult

_SYNTHESIZE_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


class GoogleTTSProvider:
    name = "google"

    def __init__(self, voice: str = GOOGLE_TTS_VOICE, language_code: str = GOOGLE_TTS_LANGUAGE_CODE):
        self.voice = voice
        self.model = voice  # Google names a voice, not a separate model
        self.language_code = language_code
        if not GCP_TTS_API_KEY:
            raise RuntimeError(
                "GCP_TTS_API_KEY (or lowercase gcp_tts_api_key) is not set — "
                "cannot construct GoogleTTSProvider. Either set it, or leave "
                "TTS_PROVIDER at its default of 'openai'."
            )
        self._api_key: str = GCP_TTS_API_KEY

    async def synthesize(self, text: str) -> SynthesisResult:
        started = time.perf_counter()
        payload = {
            "input": {"text": text},
            "voice": {"languageCode": self.language_code, "name": self.voice},
            # LINEAR16 comes back inside a WAV container, so this returns
            # the same self-describing media type the OpenAI provider does
            # and the client needs no per-provider branch.
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                _SYNTHESIZE_URL,
                headers={"X-Goog-Api-Key": self._api_key},
                json=payload,
            )
        resp.raise_for_status()
        audio = base64.b64decode(resp.json()["audioContent"])
        latency_ms = (time.perf_counter() - started) * 1000.0
        return SynthesisResult(
            audio=audio,
            media_type="audio/wav",
            latency_ms=latency_ms,
            provider=self.name,
            voice=self.voice,
            model=self.model,
        )
