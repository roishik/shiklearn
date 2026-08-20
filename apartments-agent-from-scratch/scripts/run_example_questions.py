"""Run the brief's four example questions end-to-end and save transcripts."""
import json, sys, time, datetime, pathlib
sys.path.insert(0, '.')
from app.agent_loop import run_agent, MaxTurnsExceeded
from app.providers.llm import get_llm_provider
from app.system_prompt import BASE_SYSTEM_PROMPT
from app.tools import TOOL_SCHEMAS, TOOL_REGISTRY

QUESTIONS = [
    ("Q1", "Which airports in New England are strong candidates for terminal expansion?"),
    ("Q2", "Compare LA and Santa Ana airport congestion levels."),
    ("Q3", "What is the percentage of long haul flights out of Anchorage?"),
    ("Q4", "What is the unmet flight demand in SFO and why?"),
]

provider = get_llm_provider()
print(f"provider={provider.name} model={provider.model}\n")
out = []
for qid, q in QUESTIONS:
    print(f"{'='*72}\n{qid}: {q}\n{'='*72}")
    t0 = time.time()
    try:
        r = run_agent(user_message=q, history=[], provider=provider,
                      tool_schemas=TOOL_SCHEMAS, tool_registry=TOOL_REGISTRY,
                      system_prompt=BASE_SYSTEM_PROMPT, max_turns=8)
        final, log, turns, incomplete = r.final_text, r.tool_log, r.turns_used, False
    except MaxTurnsExceeded as e:
        final, log, turns, incomplete = "(hit max_turns)", e.tool_log, e.turns_used, True
    dt = time.time() - t0
    for e in log:
        err = f"  ERROR: {e.error}" if e.error else ""
        print(f"  -> {e.tool_name}({json.dumps(e.arguments)[:110]}){err}")
    print(f"\n{final}\n[{turns} turns, {dt:.1f}s, incomplete={incomplete}]\n")
    out.append({"id": qid, "question": q, "answer": final, "turns": turns,
                "seconds": round(dt, 1), "incomplete": incomplete,
                "tool_calls": [{"tool": e.tool_name, "arguments": e.arguments,
                                "error": e.error, "result": e.result} for e in log]})

pathlib.Path("artifacts").mkdir(exist_ok=True)
payload = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "provider": provider.name, "model": provider.model, "transcripts": out}
pathlib.Path("artifacts/example_questions.json").write_text(json.dumps(payload, indent=2, default=str))
print("wrote artifacts/example_questions.json")
