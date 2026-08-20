"""
Config / env loading.

The AI_PROVIDER pattern: every value comes from an environment
variable with a sane default, and a `*_PROVIDER` string selects which
implementation a factory function returns (see
app/providers/llm/__init__.py). Nothing outside app/providers/* imports a
provider SDK or endpoint directly — that's the whole point of the
abstraction, and it's what makes "swap the model" a one-line change.

Default LLM_PROVIDER is "mock", NOT "openai". Cloning this repo and
running `python -m app.cli` or `pytest` must work with ZERO setup — no
.env file, no API key — so a reviewer is never blocked on credentials.
Set LLM_PROVIDER=openai (or =anthropic) plus the matching API key to wire
up a real model; nothing else in the app changes (see README "Swapping a
provider").

Two env-loading conveniences, both deliberate. (1) Key names are accepted
in either case (`OPENAI_API_KEY` or `openai_api_key`), because real
secrets files disagree about this and a silent case mismatch is a
miserable thing to debug. (2) A `.env` at a workspace root ABOVE this
project is honoured as a fallback, so the repo works unchanged whether
it's checked out standalone or inside a larger workspace. Load order is
project-local first, workspace second, and neither ever overwrites a var
that's already set — first loader wins.

IMPORTANT: this module must never print, log, or repr() an API key value.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

# Defense in depth against credential leakage into logs: httpx's own
# logger prints full request lines (incl. any query-string credentials)
# at INFO level. Every provider here authenticates via headers, never
# query params, but silencing httpx's logger at import time — in the one
# module every entrypoint loads first — is a second line of defense.
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("config")

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent  # the repo root


def _find_shared_env(start: Path, max_levels: int = 5) -> Path | None:
    """Walk upward from the project dir looking for a workspace-level
    .env. Deliberately NOT hardcoded to "one directory up": that
    assumption breaks the moment the repo is checked out one level deeper
    than expected, and it breaks SILENTLY — a missing shared .env is a
    harmless no-op by design, so the only symptom is the provider raising
    "no key set" several frames away with nothing pointing at the cause.
    A bounded upward walk finds it regardless of checkout depth, and the
    bound stops it escaping to the filesystem root."""
    current = start
    for _ in range(max_levels):
        current = current.parent
        candidate = current / ".env"
        if candidate.exists():
            return candidate
    return None


def _load_dotenv(path: Path) -> None:
    """Minimal .env parser (zero external dependency — no python-dotenv).
    Never overwrites a var that's already set, so first loader wins."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load order: project-local .env first (dev overrides), then a
# workspace-level .env as fallback, wherever it's found above this repo.
_load_dotenv(PROJECT_DIR / ".env")
_shared_env = _find_shared_env(PROJECT_DIR)
if _shared_env is not None:
    _load_dotenv(_shared_env)
    # Path only, never values — this line is the difference between "no key
    # set" being a 30-second fix and a 20-minute hunt.
    logger.debug("loaded shared .env from %s", _shared_env)
else:
    logger.debug("no shared .env found within %d levels above %s", 5, PROJECT_DIR)


def _env(*names: str, default: str | None = None) -> str | None:
    """Return the first set env var among `names` (case-alias support)."""
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


# ── Secrets (values never logged) ──────────────────────────────────────────
OPENAI_API_KEY = _env("OPENAI_API_KEY", "openai_api_key")
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "anthropic_api_key")
# Groq: a genuinely free tier, no credit card required — the provider a
# reviewer with no paid key can actually use to see the real tool-calling
# agent run, not just LLM_PROVIDER=mock's scripted stand-in. See
# app/providers/llm/groq_llm.py.
GROQ_API_KEY = _env("GROQ_API_KEY", "groq_api_key")

# ── Provider selection ──────────────────────────────────────────────────────
# Swapping the model is changing this one line (+ credentials) — never
# touching app/agent_loop.py, app/main.py, or app/cli.py. See README
# "Swapping a provider" and DESIGN_DOC.md §"Where and how AI is used".
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "mock")  # mock | openai | anthropic | groq

# ── Model choices ────────────────────────────────────────────────────────
OPENAI_LLM_MODEL = os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")
ANTHROPIC_LLM_MODEL = os.environ.get("ANTHROPIC_LLM_MODEL", "claude-haiku-4-5-20251001")
# Llama 3.3 70B: Groq's largest generally-available free-tier model with
# native tool calling, which this app depends on — smaller free models
# are more prone to malformed tool-call arguments.
GROQ_LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

# ── Voice (optional) ────────────────────────────────────────────────────
# The chat interface is the deliverable; voice is a bonus mode layered on
# top of it. Everything below is inert unless the UI's conversation mode
# is used, and the app starts, serves, and passes its tests with none of
# it configured.
#
# Both default providers are OpenAI, and that is a deliberate cost-of-entry
# decision rather than a preference: the app already requires an OpenAI key
# to run against a real model, so speech in and speech out add ZERO new
# credentials. A reviewer who can run the agent at all can also talk to it.
# Google TTS is implemented too (app/providers/tts/google_tts.py) and is a
# one-line switch — it exists to make "swap the vendor" a demonstrated
# property rather than a claim, since a provider interface with exactly one
# implementation has never actually been tested as an interface.
STT_PROVIDER = os.environ.get("STT_PROVIDER", "openai")  # openai
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "openai")  # openai | google

# gpt-4o-mini-transcribe over whisper-1: same family, cheaper per minute,
# and better on proper nouns — which is the whole problem in this domain,
# where nearly every user turn contains a locality or neighborhood name
# (Hebrew place names transliterated into English are easy to mis-hear).
# whisper-1 is the fallback because it is the most universally available
# transcription model on the API, so a key whose tier lacks the newer
# model still gets working speech instead of an error.
OPENAI_STT_MODEL = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
OPENAI_STT_FALLBACK_MODEL = os.environ.get("OPENAI_STT_FALLBACK_MODEL", "whisper-1")

OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "alloy")

# Google Cloud Text-to-Speech, if TTS_PROVIDER=google. Neural2 rather than
# Studio or Chirp3-HD: a clear step up from Standard without the premium
# tiers' pricing, which is the right point on the curve for a demo.
GCP_TTS_API_KEY = _env("GCP_TTS_API_KEY", "gcp_tts_api_key")
GOOGLE_TTS_VOICE = os.environ.get("GOOGLE_TTS_VOICE", "en-US-Neural2-C")
GOOGLE_TTS_LANGUAGE_CODE = os.environ.get("GOOGLE_TTS_LANGUAGE_CODE", "en-US")

# Upper bound on a single synthesis request. Not a security boundary —
# this server is a local demo — but an accident bound: nothing should be
# able to turn one stray reply into an unbounded, billable TTS call.
TTS_MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "4000"))

# Upper bound on one uploaded utterance, in bytes. At 16 kHz mono PCM16
# (what the browser client sends) 8 MB is a little over four minutes of
# audio — far past any real spoken question, and short of anything that
# would tie up the process.
STT_MAX_UPLOAD_BYTES = int(os.environ.get("STT_MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))


def have_openai_key() -> bool:
    return bool(OPENAI_API_KEY)


def have_anthropic_key() -> bool:
    return bool(ANTHROPIC_API_KEY)


def have_groq_key() -> bool:
    return bool(GROQ_API_KEY)


def have_gcp_tts_key() -> bool:
    return bool(GCP_TTS_API_KEY)
