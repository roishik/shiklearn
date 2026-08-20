"""End-to-end smoke test — not part of the pytest suite, run manually:

    python scripts/smoke_test.py            # uses LLM_PROVIDER from env/.env (default: mock)
    LLM_PROVIDER=openai python scripts/smoke_test.py   # exercises a real provider + real key

Exercises the full stack in one process: config -> provider factory ->
agent_loop -> tools -> scoring -> guardrails, with no framework and no
mocking of anything except (optionally) the LLM API call itself. Prints
enough to eyeball that tool-calling, deterministic scoring, and the
untrusted-data wrapping are all actually wired together, not just unit
tested in isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent_loop import run_agent  # noqa: E402
from app.config import LLM_PROVIDER  # noqa: E402
from app.providers.llm import get_llm_provider  # noqa: E402
from app.system_prompt import BASE_SYSTEM_PROMPT  # noqa: E402
from app.tools import TOOL_REGISTRY, TOOL_SCHEMAS  # noqa: E402


def main() -> None:
    provider = get_llm_provider()
    print(f"LLM_PROVIDER={LLM_PROVIDER!r} -> provider={provider.name} model={provider.model}\n")

    result = run_agent(
        user_message="Compare LAX and SNA congestion levels and tell me which is more constrained, and why.",
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
