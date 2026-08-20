"""The voice half of the web layer: transcribe in, synthesize out, and
the one history operation barge-in needs.

WHERE THE WORK IS SPLIT, and why it is split there.

The browser owns the microphone, endpointing, and playback. The server
owns transcription, synthesis, and the conversation. That division is not
the obvious one — the more common design runs a WebSocket, streams raw
audio up continuously, and runs voice-activity detection server-side. It
was rejected here for three concrete reasons:

  1. BARGE-IN GETS FASTER, not slower. The thing that must happen the
     instant a user talks over the agent is "stop playing audio". The
     browser is where the audio is playing and where the microphone is,
     so it already knows both facts. Doing it locally costs zero network
     round-trips; routing the decision through a server adds a round-trip
     to the one interaction where latency is most audible.
  2. NOTHING IS UPLOADED DURING SILENCE. Detecting the end of an
     utterance locally means one HTTP request per utterance instead of a
     continuously open socket carrying mostly nothing.
  3. NO NEW DEPENDENCIES. Server-side VAD wants numpy for frame energy.
     This project ships four packages, and "you can read every line that
     produces a number" is easier to defend when the list stays short.

What the server gives up by not holding the audio: it cannot do
server-side turn detection, and it cannot see the waveform for
diagnostics. Both are real, and both are the right trade at this size.

The transcript goes through EXACTLY the same agent path as typed text —
same tools, same guardrails, same deterministic scoring. Voice is an
input and output modality here, not a second brain. That is why there is
no /voice/chat endpoint: the client posts the transcript to the existing
/chat/stream and gets the existing SSE tool log, so anything true of the
text agent stays true when you talk to it.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import conversation
from app.config import (
    STT_MAX_UPLOAD_BYTES,
    STT_PROVIDER,
    TTS_MAX_CHARS,
    TTS_PROVIDER,
    have_gcp_tts_key,
    have_openai_key,
)

logger = logging.getLogger("voice")

router = APIRouter(prefix="/voice", tags=["voice"])

# Audio container types the transcription endpoint accepts. The browser
# client sends WAV, but the allowlist is broader because a different
# client (or a curl one-liner during debugging) reasonably would not.
_ACCEPTED_AUDIO_TYPES = {
    "audio/wav": "utterance.wav",
    "audio/x-wav": "utterance.wav",
    "audio/wave": "utterance.wav",
    "audio/webm": "utterance.webm",
    "audio/ogg": "utterance.ogg",
    "audio/mpeg": "utterance.mp3",
    "audio/mp4": "utterance.mp4",
    "audio/m4a": "utterance.m4a",
    "audio/flac": "utterance.flac",
}


class SpeakRequest(BaseModel):
    text: str


class InterruptRequest(BaseModel):
    # Everything the user actually heard before talking over the agent.
    # Empty is meaningful and not an error: it means they interrupted
    # before any audio played. Bounded, because it is only ever a prefix
    # of a reply that was itself bounded by TTS_MAX_CHARS.
    spoken_prefix: str = Field(default="", max_length=TTS_MAX_CHARS)


def _missing_credential() -> str | None:
    """Why voice is unavailable, in a sentence a reviewer can act on, or
    None if it is available. Checked without constructing a provider,
    because constructing one raises and this is the endpoint whose whole
    job is to answer the question calmly."""
    if STT_PROVIDER == "openai" and not have_openai_key():
        return "OPENAI_API_KEY is not set — speech-to-text needs it."
    if TTS_PROVIDER == "openai" and not have_openai_key():
        return "OPENAI_API_KEY is not set — text-to-speech needs it."
    if TTS_PROVIDER == "google" and not have_gcp_tts_key():
        return "GCP_TTS_API_KEY is not set — TTS_PROVIDER is 'google'."
    return None


@router.get("/health")
async def voice_health() -> dict:
    """Whether voice mode can work, and if not, why.

    The UI calls this before offering the control at all. A disabled
    button with a real reason on it beats a live button that fails on
    click — this app is run by people who did not configure it, and the
    difference between "voice needs a key you have not set" and a 500 in
    a console they are not watching is the entire user experience of the
    bonus feature.
    """
    reason = _missing_credential()
    return {
        "available": reason is None,
        "reason": reason,
        "stt": {"provider": STT_PROVIDER},
        "tts": {"provider": TTS_PROVIDER},
    }


@router.post("/transcribe")
async def transcribe(request: Request) -> dict:
    """One complete utterance in, text out.

    Takes the audio as the raw request body rather than a multipart form.
    Multipart would need `python-multipart`, a fifth dependency, to move
    exactly one file per request that has no accompanying fields — the
    content type header already says everything the form would have.
    """
    reason = _missing_credential()
    if reason is not None:
        raise HTTPException(status_code=503, detail=reason)

    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    filename = _ACCEPTED_AUDIO_TYPES.get(content_type)
    if filename is None:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio content-type {content_type!r}. "
            f"Expected one of: {', '.join(sorted(_ACCEPTED_AUDIO_TYPES))}.",
        )

    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio body.")
    if len(audio) > STT_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Utterance is {len(audio)} bytes, over the {STT_MAX_UPLOAD_BYTES}-byte limit.",
        )

    from app.providers.stt import get_stt_provider

    try:
        # Inside the try, not above it: an unrecognized STT_PROVIDER makes
        # the factory raise, and _missing_credential has no branch for a
        # provider it does not know, so the 503 gate above lets it
        # through. Outside the try that surfaced as an unhandled 500.
        provider = get_stt_provider()
        result = await provider.transcribe(audio, filename=filename, content_type=content_type)
    except Exception as exc:
        # Never let a provider error reach the browser verbatim: an
        # upstream 4xx body can echo request details back, and this one
        # carried an API key in its Authorization header. Log the type,
        # return a flat message.
        logger.warning("transcription failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Transcription failed upstream.") from exc

    return {
        # Stripped here as well as in the provider. A provider is an
        # external contract, and the caller decides on this string
        # ("is there anything here at all?") — that decision should not
        # depend on which vendor happens to be configured.
        "text": result.text.strip(),
        "latency_ms": round(result.latency_ms, 1),
        "provider": result.provider,
        "model": result.model,
    }


@router.post("/speak")
async def speak(req: SpeakRequest) -> Response:
    """One sentence in, one audio file out.

    A sentence, not a whole reply, and that granularity is load-bearing
    rather than an arbitrary chunk size. It is what lets the first words
    start playing while the rest is still being synthesized, and it is
    what makes barge-in truncation honest: history can only be rewritten
    to "what the user heard" if the units that were played match the
    units that were requested. See app/conversation.py.
    """
    reason = _missing_credential()
    if reason is not None:
        raise HTTPException(status_code=503, detail=reason)

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Nothing to speak.")
    if len(text) > TTS_MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Text is {len(text)} characters, over the {TTS_MAX_CHARS}-character limit.",
        )

    from app.providers.tts import get_tts_provider

    try:
        # Inside the try for the same reason as /transcribe — an unknown
        # TTS_PROVIDER must be a handled error, not a 500.
        provider = get_tts_provider()
        result = await provider.synthesize(text)
    except Exception as exc:
        logger.warning("synthesis failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Speech synthesis failed upstream.") from exc

    return Response(
        content=result.audio,
        media_type=result.media_type,
        headers={
            # Diagnostics on the response rather than in a second call:
            # the client shows provider and latency in the UI, and asking
            # for them separately would have doubled the round-trips on
            # the latency-sensitive path.
            "X-Voice-Provider": result.provider,
            "X-Voice-Voice": result.voice,
            "X-Voice-Model": result.model,
            "X-Voice-Latency-Ms": str(round(result.latency_ms, 1)),
            "Cache-Control": "no-store",
        },
    )


@router.post("/interrupt")
async def interrupt(req: InterruptRequest) -> dict:
    """Barge-in, step three: make the transcript match what was heard.

    Steps one and two happen entirely in the browser — stop playback,
    drop what is queued — because that is where the audio is. This is the
    step that cannot happen there, and the step that is easy to skip: if
    the agent was cut off four sentences into nine, the model must not
    believe it said the other five, or every following turn is reasoning
    about a conversation that did not occur.
    """
    truncated = conversation.truncate_last_reply(req.spoken_prefix)
    # False covers two different situations and neither is an error: there
    # was no reply to cut (the user spoke over the thinking pause), or the
    # text offered was not actually the opening of the stored reply, in
    # which case it is refused rather than written. See
    # app/conversation.py — this endpoint can only ever shorten a reply,
    # never add to one.
    return {"truncated": truncated}
