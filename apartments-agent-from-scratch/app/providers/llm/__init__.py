"""LLM provider factory — the AI_PROVIDER pattern.

Default is `mock` (see app/config.py) so the whole app runs with
zero setup. Swap by setting LLM_PROVIDER=openai, =anthropic, or =groq in
.env — nothing outside this file and app/config.py needs to change.
`groq` is the free-tier option: no credit card required, for a reviewer
who wants to see the real tool-calling agent run without a paid key."""
from __future__ import annotations

from app.config import LLM_PROVIDER
from app.providers.llm.base import LLMProvider, LLMResponse, ToolCall  # noqa: F401

_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is not None:
        return _provider

    if LLM_PROVIDER == "mock":
        from app.providers.llm.mock_llm import MockLLMProvider

        _provider = MockLLMProvider()
    elif LLM_PROVIDER == "openai":
        from app.providers.llm.openai_llm import OpenAILLMProvider

        _provider = OpenAILLMProvider()
    elif LLM_PROVIDER == "anthropic":
        from app.providers.llm.anthropic_llm import AnthropicLLMProvider

        _provider = AnthropicLLMProvider()
    elif LLM_PROVIDER == "groq":
        from app.providers.llm.groq_llm import GroqLLMProvider

        _provider = GroqLLMProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Use 'mock', 'openai', 'anthropic', or 'groq'.")
    return _provider


def reset_provider_cache() -> None:
    """Test helper — clears the memoized singleton so tests can swap
    providers between cases without process restart."""
    global _provider
    _provider = None
