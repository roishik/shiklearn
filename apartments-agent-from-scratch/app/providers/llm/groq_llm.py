"""Groq LLM provider — the free-tier swap target for LLM_PROVIDER=groq.

Groq's Chat Completions endpoint is wire-compatible with OpenAI's (same
request/response shape, same tool-calling protocol), so this is
deliberately a thin subclass of OpenAILLMProvider rather than a second
copy of the request/response plumbing — only the base URL, default
model, and key lookup differ. If Groq's API ever diverges from OpenAI's
shape, override chat() here rather than letting the subclass silently
drift out of sync with its parent.

WHY THIS EXISTS: a free-tier provider is an accepted swap target, and
without one, a reviewer with no paid OpenAI/Anthropic key cannot see the
real tool-calling agent run at all — only
LLM_PROVIDER=mock's scripted stand-in. Groq's free tier needs no credit
card, has generous rate limits for a quick evaluation session, and
serves several open-weight models (Llama 3.3, GPT-OSS) that support
native tool calling, which this app depends on.

NOTE: verified in this build only via the OpenAI provider; this file has
not been exercised against a live Groq key (creating an account is not
something automated here — see DECISIONS.md). Same caveat as
anthropic_llm.py: if swapping LLM_PROVIDER=groq requires touching
agent_loop.py, app/main.py, or app/cli.py, the abstraction has a leak."""
from __future__ import annotations

from app.config import GROQ_API_KEY, GROQ_LLM_MODEL
from app.providers.llm.openai_llm import OpenAILLMProvider

_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLMProvider(OpenAILLMProvider):
    name = "groq"

    def __init__(self, model: str = GROQ_LLM_MODEL):
        # Deliberately NOT calling OpenAILLMProvider.__init__: it checks
        # OPENAI_API_KEY, which need not be set for a Groq-only reviewer.
        # Set the same attributes it would have set, against Groq's key
        # and endpoint instead — self._chat_url is what makes the
        # inherited chat() hit Groq rather than OpenAI.
        self.model = model
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY (or lowercase groq_api_key) is not set — cannot "
                "construct GroqLLMProvider. Get a free key at "
                "https://console.groq.com/keys (no credit card required), or "
                "use LLM_PROVIDER=mock to run without any key."
            )
        self._api_key: str = GROQ_API_KEY
        self._chat_url: str = _CHAT_URL
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
