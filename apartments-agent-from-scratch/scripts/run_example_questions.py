"""
run_example_questions.py — run this domain's question shapes end to end
through the REAL agent loop (app.agent_loop.run_agent, the same function
app/main.py and app/cli.py call), printing question -> each tool call ->
final answer, and saving a transcript to artifacts/example_questions.json.

ONE-OFF UTILITY, NOT A TEST. Not part of the pytest suite (pytest.ini's
testpaths is "tests" only, so this file is never collected). Run it by
hand:

    PYTHONPATH=. python scripts/run_example_questions.py

Works with zero keys and zero network on LLM_PROVIDER=mock (the default —
see app/config.py), and unmodified with a real provider once a key is
configured (LLM_PROVIDER=openai/anthropic/groq plus the matching
*_API_KEY) — same run_agent() call either way, no mock-specific branching
in the question-running path below. The provider actually in use is
printed up front; no key VALUE is ever printed or logged (app/config.py's
own rule, honoured here too).

QUESTIONS. The four shapes from _working/PROGRESS.md's table, plus two
more asked for by name: a "Compare Modi'in and Ramat Gan" question (must
surface resolve_entity's decisive=false handling for Modi'in, which is
genuinely ambiguous between Modi'in-Makkabbim-Re'ut id 1200 and Modi'in
Illit id 3797 — see app/dataset.py / _working/PROGRESS.md) and a Hebrew
question ("compare Ra'anana and Kefar Sava").

KNOWN, DOCUMENTED LIMITATION OF LLM_PROVIDER=mock (see
app/providers/llm/mock_llm.py's own module docstring): it is a SCRIPTED
two-phase stand-in that ALWAYS requests `compare_items` on turn 1,
regardless of what the question actually asks — it never calls
find_items, resolve_entity, aggregate_records, estimate_derived_metric,
rank_by_priorities, or any other tool on its own. Under mock, every
question below will therefore show a compare_items call, even the ones
whose real intent is a different tool (the aggregate-share question, the
derived-income question, the ambiguous-compare question). That is not a
bug in this script or in the agent loop; it is exactly what the mock
provider is (see its docstring for why it exists at all — offline,
zero-setup, zero-network runnability). This script prints that caveat
once, up front, rather than letting a mock run silently imply it
demonstrated tool ROUTING it did not.

Because of that limitation, this script makes ONE extra, ALWAYS-RUN call
straight to TOOL_REGISTRY["resolve_entity"] for "Modi'in", OUTSIDE the
agent loop, clearly labeled as a direct-tool demonstration — this is what
actually proves decisive=false works for the ambiguous-Modi'in case
regardless of which provider is configured, since mock's scripted
behavior would otherwise never call resolve_entity at all.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agent_loop import MaxTurnsExceeded, run_agent  # noqa: E402
from app.config import LLM_PROVIDER, have_anthropic_key, have_groq_key, have_openai_key  # noqa: E402
from app.providers.llm import get_llm_provider  # noqa: E402
from app.system_prompt import BASE_SYSTEM_PROMPT  # noqa: E402
from app.tools import TOOL_REGISTRY, TOOL_SCHEMAS  # noqa: E402

QUESTIONS: list[tuple[str, str]] = [
    # Shape 1 — filtered ranking: find_items -> compare_items.
    ("Q1_filtered_ranking", "Which places up north are good value under 2.2M shekels?"),
    # Shape 2 — compare, unambiguous names: resolve_entity -> compare_items.
    ("Q2_compare_unambiguous", "Compare Kfar Saba and Ra'anana and tell me which is the better value, and why."),
    # Shape 3 — single-entity aggregate: aggregate_records, NOT compare_items.
    ("Q3_single_entity_aggregate", "What share of Tel Aviv's neighborhoods are priced above the city median?"),
    # Shape 4 — derived + causal: estimate_derived_metric.
    ("Q4_derived_causal", "What monthly household income do I need to buy a median 4-room flat in Be'er Sheva, and why?"),
    # Extra 1 — ambiguous compare: resolve_entity must come back decisive=false
    # for "Modi'in" (1200 Modi'in-Makkabbim-Re'ut vs 3797 Modi'in Illit).
    ("Q5_ambiguous_compare", "Compare Modi'in and Ramat Gan."),
    # Extra 2 — Hebrew: "Compare Ra'anana and Kefar Sava" in Hebrew.
    ("Q6_hebrew", "השווה בין רעננה לכפר סבא"),
]


def _provider_summary() -> str:
    """Which provider is actually in use, and whether a key is configured
    for it — never the key's VALUE. Printed unconditionally so a reader
    of the console output (or the saved transcript) can tell at a glance
    whether this was a mock run or a real one."""
    # `mock` is deliberately NOT reported as "key configured: True". It
    # needs no key by construction, so the old phrasing was technically
    # true and practically backwards: it printed the reassuring line
    # exactly on the run where no real model is involved at all, which is
    # the one case this function exists to make unmistakable.
    if LLM_PROVIDER == "mock":
        return (
            f"LLM_PROVIDER={LLM_PROVIDER!r} — NO REAL MODEL. "
            "Scripted stub, no key needed and none used."
        )
    key_status = {
        "openai": have_openai_key(),
        "anthropic": have_anthropic_key(),
        "groq": have_groq_key(),
    }
    have_key = key_status.get(LLM_PROVIDER, False)
    if not have_key:
        return (
            f"LLM_PROVIDER={LLM_PROVIDER!r} but NO KEY IS CONFIGURED for it — "
            "calls will fail. Set it in .env, or use LLM_PROVIDER=mock."
        )
    return f"LLM_PROVIDER={LLM_PROVIDER!r}, real key configured — this is a live run."


def _run_one(qid: str, question: str, provider) -> dict:
    print(f"\n{'=' * 78}\n{qid}: {question}\n{'=' * 78}")
    started = time.time()
    try:
        result = run_agent(
            user_message=question,
            history=[],
            provider=provider,
            tool_schemas=TOOL_SCHEMAS,
            tool_registry=TOOL_REGISTRY,
            system_prompt=BASE_SYSTEM_PROMPT,
            max_turns=8,
        )
        final_text, tool_log, turns_used, incomplete = result.final_text, result.tool_log, result.turns_used, False
    except MaxTurnsExceeded as exc:
        final_text, tool_log, turns_used, incomplete = "(hit max_turns without a final answer)", exc.tool_log, exc.turns_used, True
    elapsed = time.time() - started

    for entry in tool_log:
        status = f"  ERROR: {entry.error}" if entry.error else ""
        print(f"  -> {entry.tool_name}({json.dumps(entry.arguments, ensure_ascii=False)[:140]}){status}")
    print(f"\n{final_text}\n[{turns_used} turn(s), {elapsed:.1f}s, incomplete={incomplete}]")

    return {
        "id": qid,
        "question": question,
        "final_answer": final_text,
        "turns_used": turns_used,
        "seconds": round(elapsed, 2),
        "incomplete": incomplete,
        "tool_calls": [
            {"tool": e.tool_name, "arguments": e.arguments, "error": e.error, "result": e.result}
            for e in tool_log
        ],
    }


def _run_resolve_entity_demo() -> dict:
    """ALWAYS run, regardless of provider — see module docstring. Proves
    decisive=false actually fires for the genuinely ambiguous 'Modiin'
    query, by calling the real tool function directly rather than relying
    on an LLM (mock or otherwise) to choose to call it.

    SPELLING MATTERS HERE, and it's worth being explicit about why: only
    the bare 'Modiin' (no apostrophe) is non-decisive. Verified directly
    against the real tool while writing this script:
      resolve_entity('Modiin')   -> decisive=False, 1200 @ 0.8581 vs 3797 @ 0.8227 (gap 0.0354)
      resolve_entity("Modi'in")  -> decisive=True,  1200 @ 1.0    vs 3797 @ 0.839  (gap 0.161)
      resolve_entity('מודיעין')  -> decisive=False, 3797 @ 0.9223 vs 1200 @ 0.8967 (gap 0.0256)
    "Modi'in" (with the apostrophe) is decisive because it exact-matches
    a curated alias that belongs specifically to 1200
    (Modi'in-Makkabbim-Re'ut's own name), clearing DECISIVE_MIN_GAP=0.15
    against the runner-up. 'Modiin' has no such exact alias for either
    locality, so it falls to fuzzy matching, where the two candidates are
    close enough (gap 0.0354, well under the 0.15 bar) to correctly stay
    non-decisive — which is the real ambiguity this domain has (see
    _working/PROGRESS.md). Using 'Modiin' here, not "Modi'in", is
    therefore deliberate, not a typo.
    """
    query = "Modiin"
    print(f"\n{'=' * 78}\nDIRECT TOOL DEMO (not via the agent loop): resolve_entity({query!r})\n{'=' * 78}")
    result = TOOL_REGISTRY["resolve_entity"]({"query": query})
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    assert result["decisive"] is False, (
        f"expected resolve_entity({query!r}) to be non-decisive (1200 Modi'in-Makkabbim-Re'ut "
        "vs 3797 Modi'in Illit are both real, competing candidates) -- if this assertion ever "
        "fires, something about the dataset or entity_resolution.py changed; investigate before "
        "trusting this script's other output."
    )
    print("\nconfirmed: decisive=false, as expected for a genuinely ambiguous locality prefix.")
    return result


def main() -> None:
    provider = get_llm_provider()
    print(f"provider={provider.name} model={provider.model} | {_provider_summary()}")
    print(
        "\nNOTE: LLM_PROVIDER=mock is a SCRIPTED two-phase stand-in (see "
        "app/providers/llm/mock_llm.py) that ALWAYS requests compare_items on turn 1, "
        "regardless of the question's actual intent. Under mock, every question below will "
        "show a compare_items call even where the real tool-routing answer is find_items, "
        "aggregate_records, estimate_derived_metric, or resolve_entity -- that is a known "
        "limitation of the mock provider, not a bug in the agent loop or this script. Re-run "
        "with a real provider (LLM_PROVIDER=openai/anthropic/groq plus a key) to see genuine "
        "tool routing."
        if provider.name == "mock"
        else "\nRunning against a REAL provider -- tool routing below reflects genuine model choices."
    )

    transcripts = [_run_one(qid, question, provider) for qid, question in QUESTIONS]
    resolve_entity_demo = _run_resolve_entity_demo()

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "provider": provider.name,
        "model": provider.model,
        "questions": transcripts,
        "resolve_entity_direct_demo": resolve_entity_demo,
    }
    out_path = out_dir / "example_questions.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
