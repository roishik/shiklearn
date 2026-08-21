"""End-to-end smoke test for the Israel home-buying agent — not part of
the pytest suite (pytest.ini's testpaths is "tests" only, so this file is
never collected), run manually:

    python scripts/smoke_test.py            # uses LLM_PROVIDER from env/.env (default: mock)
    LLM_PROVIDER=openai python scripts/smoke_test.py   # exercises a real provider + real key

Exercises the full stack in one process: config -> provider factory ->
agent_loop -> tools -> scoring -> guardrails, with no framework and no
mocking of anything except (optionally) the LLM API call itself. Prints
enough to eyeball that tool-calling, deterministic scoring, and the
untrusted-data wrapping are all actually wired together, not just unit
tested in isolation.

FAILS LOUDLY WHEN NO REAL KEY IS CONFIGURED for whatever LLM_PROVIDER is
requested (openai/anthropic/groq) — this is a MANUAL check meant to
exercise a real model end to end, so silently falling back to the mock
provider and reporting success would defeat the entire point. Only when
LLM_PROVIDER is left at its default ('mock') does it run against the mock
provider on purpose, and it says so plainly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent_loop import run_agent  # noqa: E402
from app.config import LLM_PROVIDER, have_anthropic_key, have_groq_key, have_openai_key  # noqa: E402
from app.providers.llm import get_llm_provider  # noqa: E402
from app.system_prompt import BASE_SYSTEM_PROMPT  # noqa: E402
from app.tools import TOOL_REGISTRY, TOOL_SCHEMAS  # noqa: E402

_HAVE_KEY = {"openai": have_openai_key, "anthropic": have_anthropic_key, "groq": have_groq_key}


def _fail_loudly_if_no_key() -> None:
    """A real-provider request with no matching key is a MISCONFIGURATION
    for this specific script's purpose (it exists to exercise a real
    model), not something to paper over by silently running the mock
    provider instead and reporting a green smoke test that tested nothing
    of the kind. mock itself needs no key and is skipped here."""
    if LLM_PROVIDER == "mock":
        print(f"LLM_PROVIDER={LLM_PROVIDER!r} -- running against the mock provider on purpose (no key needed).\n")
        return
    checker = _HAVE_KEY.get(LLM_PROVIDER)
    if checker is None:
        print(f"FATAL: LLM_PROVIDER={LLM_PROVIDER!r} is not one of mock/openai/anthropic/groq.", file=sys.stderr)
        raise SystemExit(1)
    if not checker():
        print(
            f"FATAL: LLM_PROVIDER={LLM_PROVIDER!r} but no matching API key is configured "
            f"(see app/config.py have_{LLM_PROVIDER}_key()). This script exists to exercise a "
            "REAL provider end to end -- set the key (e.g. in a .env file) or unset "
            "LLM_PROVIDER to fall back to the mock provider on purpose.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> None:
    _fail_loudly_if_no_key()
    provider = get_llm_provider()
    print(f"LLM_PROVIDER={LLM_PROVIDER!r} -> provider={provider.name} model={provider.model}\n")

    result = run_agent(
        user_message="Compare Kefar Sava and Ra'anana and tell me which is the better value for money, and why.",
        history=[],
        provider=provider,
        tool_schemas=TOOL_SCHEMAS,
        tool_registry=TOOL_REGISTRY,
        system_prompt=BASE_SYSTEM_PROMPT,
    )

    print(f"turns_used={result.turns_used}")
    print(f"tool calls made: {len(result.tool_log)}")
    for entry in result.tool_log:
        status = f"ERROR: {entry.error}" if entry.error else "ok"
        print(f"  - {entry.tool_name}({entry.arguments}) -> {status}")
    print("\n--- final answer ---")
    print(result.final_text)

    assert result.final_text, "smoke test failed: empty final answer"
    assert result.tool_log, "smoke test failed: agent never called a tool"
    assert all(e.error is None for e in result.tool_log), "smoke test failed: a tool call errored"
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
