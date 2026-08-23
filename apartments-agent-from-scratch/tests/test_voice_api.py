"""Tests for app/voice_api.py — the voice HTTP surface.

No network and no credentials: providers are replaced with fakes, so
these run identically on a bare clone and in a CI box with no keys. What
they cover is everything the endpoints decide BEFORE a vendor is
involved, which is where all of the interesting behaviour lives —
availability reporting, input validation, the size and type gates, and
the promise that an upstream error never reaches the browser verbatim.

That last one is worth stating: the transcription and synthesis requests
carry an API key in an Authorization header, and an upstream 4xx body can
echo request details back. Forwarding that to a browser is how keys end
up in someone's devtools. Two tests below pin that shut.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import conversation, voice_api
from app.main import app
from app.providers.stt.base import TranscriptResult
from app.providers.tts.base import SynthesisResult

TINY_WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


@pytest.fixture
def client():
    conversation.clear()
    with TestClient(app) as c:
        yield c
    conversation.clear()


@pytest.fixture
def voice_available(monkeypatch):
    """Make every credential check pass without any real key present.

    Three functions, not one: _missing_credential() is the COMBINED
    check /voice/health uses (needs STT and TTS both), while
    /voice/transcribe and /voice/speak each call only their own scoped
    half (_missing_stt_credential / _missing_tts_credential) — see
    app/voice_api.py's docstring on _missing_credential for why calling
    the combined check from either single-capability endpoint was a
    real bug. A fixture claiming "voice is available" has to make all
    three agree, or tests using it would silently stop covering the
    endpoints they claim to."""
    monkeypatch.setattr(voice_api, "_missing_credential", lambda: None)
    monkeypatch.setattr(voice_api, "_missing_stt_credential", lambda: None)
    monkeypatch.setattr(voice_api, "_missing_tts_credential", lambda: None)


class FakeSTT:
    name = "fake"
    model = "fake-stt"

    def __init__(self):
        self.calls = []

    async def transcribe(self, audio, *, filename, content_type):
        self.calls.append((audio, filename, content_type))
        return TranscriptResult(
            text="  which neighborhoods in Ra'anana are good value  ",
            latency_ms=12.345,
            provider=self.name,
            model=self.model,
        )


class FakeTTS:
    name = "fake"
    voice = "fake-voice"
    model = "fake-tts"

    def __init__(self):
        self.calls = []

    async def synthesize(self, text):
        self.calls.append(text)
        return SynthesisResult(
            audio=b"AUDIOBYTES",
            media_type="audio/wav",
            latency_ms=67.891,
            provider=self.name,
            voice=self.voice,
            model=self.model,
        )


@pytest.fixture
def fake_stt(monkeypatch):
    provider = FakeSTT()
    monkeypatch.setattr("app.providers.stt.get_stt_provider", lambda: provider)
    return provider


@pytest.fixture
def fake_tts(monkeypatch):
    provider = FakeTTS()
    monkeypatch.setattr("app.providers.tts.get_tts_provider", lambda: provider)
    return provider


# ── Availability ───────────────────────────────────────────────────────


def test_health_reports_available_when_credentials_are_present(client, voice_available):
    body = client.get("/voice/health").json()
    assert body["available"] is True
    assert body["reason"] is None
    assert "provider" in body["stt"] and "provider" in body["tts"]


def test_health_reports_the_reason_when_a_key_is_missing(client, monkeypatch):
    monkeypatch.setattr("app.config.OPENAI_API_KEY", None)
    monkeypatch.setattr(voice_api, "STT_PROVIDER", "openai")
    body = client.get("/voice/health").json()
    assert body["available"] is False
    # The reason has to name the variable — this string is the whole
    # difference between a reviewer fixing it in ten seconds and giving up.
    assert "OPENAI_API_KEY" in body["reason"]


def test_health_never_500s_without_credentials(client, monkeypatch):
    # The endpoint whose job is to answer "can voice work" must not need
    # voice to work in order to answer.
    monkeypatch.setattr("app.config.OPENAI_API_KEY", None)
    monkeypatch.setattr("app.config.GCP_TTS_API_KEY", None)
    assert client.get("/voice/health").status_code == 200


def test_endpoints_refuse_politely_when_unconfigured(client, monkeypatch):
    # Each endpoint's OWN scoped check, not the combined one -- see the
    # voice_available fixture's docstring. Patching only
    # _missing_credential here would have zero effect on either endpoint
    # and this test would pass for the wrong reason (the real, unpatched
    # checks happening to also fail in a keyless test environment).
    monkeypatch.setattr(voice_api, "_missing_stt_credential", lambda: "OPENAI_API_KEY is not set.")
    monkeypatch.setattr(voice_api, "_missing_tts_credential", lambda: "OPENAI_API_KEY is not set.")
    assert client.post("/voice/transcribe", content=TINY_WAV, headers={"Content-Type": "audio/wav"}).status_code == 503
    assert client.post("/voice/speak", json={"text": "hello"}).status_code == 503


def test_speak_works_when_only_stt_is_misconfigured(client, monkeypatch, fake_tts):
    """The real bug this pins: /voice/speak is a TEXT-TO-SPEECH endpoint
    and must not care whether speech-to-TEXT is configured. Before
    app/voice_api.py split _missing_credential() into per-capability
    halves, both /voice/transcribe and /voice/speak called the same
    combined check, which looked at STT first — so a locality with
    TTS_PROVIDER=google fully configured and working, but no
    OPENAI_API_KEY for STT, got /voice/speak wrongly 503ing and blaming
    speech-to-text for a request that never touched it."""
    monkeypatch.setattr(voice_api, "_missing_stt_credential", lambda: "OPENAI_API_KEY is not set — speech-to-text needs it.")
    monkeypatch.setattr(voice_api, "_missing_tts_credential", lambda: None)
    resp = client.post("/voice/speak", json={"text": "hello"})
    assert resp.status_code == 200


def test_transcribe_stays_blocked_when_only_stt_is_misconfigured(client, monkeypatch, fake_stt):
    """The mirror case: /voice/transcribe must still correctly refuse when
    STT itself (the capability it actually uses) is unconfigured, even if
    TTS happens to be fine — confirms the split didn't just move the bug
    from over-blocking to under-blocking."""
    monkeypatch.setattr(voice_api, "_missing_stt_credential", lambda: "OPENAI_API_KEY is not set — speech-to-text needs it.")
    monkeypatch.setattr(voice_api, "_missing_tts_credential", lambda: None)
    resp = client.post("/voice/transcribe", content=TINY_WAV, headers={"Content-Type": "audio/wav"})
    assert resp.status_code == 503


# ── Transcription ──────────────────────────────────────────────────────


def test_transcribe_returns_stripped_text_and_provider_metadata(client, voice_available, fake_stt):
    body = client.post(
        "/voice/transcribe", content=TINY_WAV, headers={"Content-Type": "audio/wav"}
    ).json()
    assert body["text"] == "which neighborhoods in Ra'anana are good value"
    assert body["provider"] == "fake"
    assert body["model"] == "fake-stt"
    assert body["latency_ms"] == 12.3


def test_transcribe_passes_a_filename_matching_the_content_type(client, voice_available, fake_stt):
    # The transcription API infers the container from the filename it is
    # given, so a mislabelled extension is a silent accuracy problem
    # rather than an error.
    client.post("/voice/transcribe", content=b"OggS...", headers={"Content-Type": "audio/ogg"})
    _, filename, content_type = fake_stt.calls[-1]
    assert filename.endswith(".ogg")
    assert content_type == "audio/ogg"


def test_transcribe_tolerates_a_charset_suffix_on_the_content_type(client, voice_available, fake_stt):
    resp = client.post(
        "/voice/transcribe", content=TINY_WAV, headers={"Content-Type": "audio/wav; charset=binary"}
    )
    assert resp.status_code == 200


def test_transcribe_rejects_a_non_audio_content_type(client, voice_available, fake_stt):
    resp = client.post("/voice/transcribe", content=b"hello", headers={"Content-Type": "text/plain"})
    assert resp.status_code == 415
    assert not fake_stt.calls


def test_transcribe_rejects_an_empty_body(client, voice_available, fake_stt):
    resp = client.post("/voice/transcribe", content=b"", headers={"Content-Type": "audio/wav"})
    assert resp.status_code == 400
    assert not fake_stt.calls


def test_transcribe_rejects_an_oversized_upload(client, voice_available, fake_stt, monkeypatch):
    monkeypatch.setattr(voice_api, "STT_MAX_UPLOAD_BYTES", 16)
    resp = client.post("/voice/transcribe", content=b"x" * 64, headers={"Content-Type": "audio/wav"})
    assert resp.status_code == 413
    assert not fake_stt.calls


def test_transcribe_does_not_leak_an_upstream_error_to_the_browser(client, voice_available, monkeypatch):
    class Exploding:
        name = "boom"
        model = "boom"

        async def transcribe(self, audio, *, filename, content_type):
            raise RuntimeError("401 from api.openai.com: Bearer sk-secret-value-here")

    monkeypatch.setattr("app.providers.stt.get_stt_provider", lambda: Exploding())
    resp = client.post("/voice/transcribe", content=TINY_WAV, headers={"Content-Type": "audio/wav"})
    assert resp.status_code == 502
    assert "sk-secret" not in resp.text
    assert resp.json()["detail"] == "Transcription failed upstream."


# ── Synthesis ──────────────────────────────────────────────────────────


def test_speak_returns_audio_with_diagnostic_headers(client, voice_available, fake_tts):
    resp = client.post("/voice/speak", json={"text": "Ra'anana leads."})
    assert resp.status_code == 200
    assert resp.content == b"AUDIOBYTES"
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.headers["x-voice-provider"] == "fake"
    assert resp.headers["x-voice-model"] == "fake-tts"
    assert resp.headers["x-voice-latency-ms"] == "67.9"
    assert resp.headers["cache-control"] == "no-store"


def test_speak_rejects_empty_and_whitespace_only_text(client, voice_available, fake_tts):
    assert client.post("/voice/speak", json={"text": ""}).status_code == 400
    assert client.post("/voice/speak", json={"text": "   "}).status_code == 400
    assert not fake_tts.calls


def test_speak_caps_the_text_length(client, voice_available, fake_tts, monkeypatch):
    # Not a security boundary — this is a local demo — but a bound on
    # accidents: nothing should turn one stray reply into an unbounded
    # billable synthesis call.
    monkeypatch.setattr(voice_api, "TTS_MAX_CHARS", 10)
    resp = client.post("/voice/speak", json={"text": "x" * 50})
    assert resp.status_code == 413
    assert not fake_tts.calls


def test_speak_does_not_leak_an_upstream_error_to_the_browser(client, voice_available, monkeypatch):
    class Exploding:
        name = "boom"
        voice = "boom"
        model = "boom"

        async def synthesize(self, text):
            raise RuntimeError("400 from api.openai.com: Bearer sk-secret-value-here")

    monkeypatch.setattr("app.providers.tts.get_tts_provider", lambda: Exploding())
    resp = client.post("/voice/speak", json={"text": "hello"})
    assert resp.status_code == 502
    assert "sk-secret" not in resp.text


# ── Barge-in ───────────────────────────────────────────────────────────


def test_interrupt_truncates_the_last_reply(client):
    conversation.replace(
        [
            {"role": "user", "content": "rank the north"},
            {"role": "assistant", "content": "Ra'anana leads. Kfar Saba follows. Hadera is third."},
        ]
    )
    body = client.post("/voice/interrupt", json={"spoken_prefix": "Ra'anana leads."}).json()
    assert body["truncated"] is True
    assert conversation.snapshot()[-1]["content"] == "Ra'anana leads."


def test_interrupt_with_nothing_to_truncate_is_not_an_error(client):
    # The user talked over the "thinking" pause, before a reply existed.
    resp = client.post("/voice/interrupt", json={"spoken_prefix": "hello"})
    assert resp.status_code == 200
    assert resp.json()["truncated"] is False


def test_interrupt_defaults_to_an_empty_prefix(client):
    conversation.replace([{"role": "assistant", "content": "Ra'anana leads."}])
    assert client.post("/voice/interrupt", json={}).json()["truncated"] is True
    assert conversation.snapshot() == []


def test_interrupt_needs_no_credentials(client, monkeypatch):
    # Barge-in touches no vendor. It must keep working even when voice is
    # half-configured, or an interrupted conversation stays corrupted.
    monkeypatch.setattr("app.config.OPENAI_API_KEY", None)
    conversation.replace([{"role": "assistant", "content": "Ra'anana leads. Kfar Saba follows."}])
    assert client.post("/voice/interrupt", json={"spoken_prefix": "Ra'anana leads."}).status_code == 200


# ── The integration promise ────────────────────────────────────────────


def test_voice_routes_do_not_disturb_the_text_interface(client):
    # The whole feature is additive. If mounting the voice router changed
    # anything about /health or /, that would be the finding.
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200
