"""Terminal chat loop — the zero-dependency chat UI. Run with:

    python -m app.cli

Works with zero setup (LLM_PROVIDER defaults to mock — see
app/config.py). Prints each tool call as it happens, so the agent's
reasoning/tool-use is visible in the terminal, same information the web
UI's tool-call log shows (app/main.py + static/index.html). Keep this
file boring on purpose — see README "UI scope — what NOT to build"."""
from __future__ import annotations

import json

from app.agent_loop import run_agent
from app.config import LLM_PROVIDER
from app.providers.llm import get_llm_provider
from app.system_prompt import BASE_SYSTEM_PROMPT
from app.tools import TOOL_REGISTRY, TOOL_SCHEMAS


def main() -> None:
    provider = get_llm_provider()
    print(f"home-buying agent chat — LLM_PROVIDER={LLM_PROVIDER} ({provider.name}/{provider.model})")
    print("Type a message, or 'exit' to quit.")
    print("Try: 'which places in the north are good value for a family with a 2.2M ₪ budget?'\n")

    history: list[dict] = []
    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        result = run_agent(
            user_message=user_message,
            history=history,
            provider=provider,
            tool_schemas=TOOL_SCHEMAS,
            tool_registry=TOOL_REGISTRY,
            system_prompt=BASE_SYSTEM_PROMPT,
        )
        for entry in result.tool_log:
            status = f"ERROR: {entry.error}" if entry.error else "ok"
            print(f"  [tool] {entry.tool_name}({json.dumps(entry.arguments)}) -> {status}")
        print(f"agent> {result.final_text}\n")
        history = result.messages[1:]  # drop the system message; run_agent re-adds it each call


if __name__ == "__main__":
    main()
