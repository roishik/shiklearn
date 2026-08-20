"""Minimal FastAPI chat UI shell.

Deliberately small: one page (static/index.html), a POST /chat/stream
endpoint (SSE), a non-streaming POST /chat kept for any caller that wants
a single JSON response, one GET /health. The optional voice routes live
next door in app/voice_api.py and are mounted here. This is the whole web
UI. See
README "UI scope — what NOT to build" for why it stops here: UI polish is
not the point of this project, and every hour spent on it is an hour not
spent on the scoring logic and the data.

SSE, not token streaming. app/providers/llm/openai_llm.py's docstring
already explains why streaming the MODEL's tokens buys nothing on a
tool-calling turn — the loop needs the complete tool_calls list before it
can act. What genuinely was missing: on a multi-tool question (several
seconds, several tool calls), the UI showed nothing at all until the
whole loop finished, then dumped the entire tool-call log at once. /chat/
stream fixes that by pushing each ToolLogEntry to the browser the instant
that tool call finishes (via agent_loop.run_agent's on_tool_call hook),
so the live tool-call log is actually live.

Run with:
    uvicorn app.main:app --reload
or:
    python -m app.main
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import conversation
from app.agent_loop import AgentResult, MaxTurnsExceeded, ToolLogEntry, run_agent
from app.config import HOST, PORT
from app.providers.llm import get_llm_provider
from app.system_prompt import BASE_SYSTEM_PROMPT
from app.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from app.voice_api import router as voice_router

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="israel-home-buying-agent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# Voice: /voice/health, /voice/transcribe, /voice/speak, /voice/interrupt.
# Additive — the text interface below does not know it exists, and every
# route here works unchanged with no speech credentials configured.
app.include_router(voice_router)

# Conversation history lives in app/conversation.py, because the voice
# endpoints share it. In-memory and process-wide: right for a single-user
# demo, wrong for multi-user. Scope note, not an oversight; see README.


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health() -> dict:
    provider = get_llm_provider()
    return {"status": "ok", "provider": provider.name, "model": provider.model}


def _tool_log_rows(entries) -> list[dict]:
    return [
        {"tool_name": e.tool_name, "arguments": e.arguments, "result": e.result, "error": e.error}
        for e in entries
    ]


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    # Deliberately `def`, not `async def`. run_agent is fully synchronous
    # and blocks on the network for as long as the whole agent turn takes
    # — the reason /chat/stream runs it on a worker thread. Declared
    # `async`, it ran that blocking loop directly on the event loop, and
    # one /chat request froze every other request in the process:
    # measured, a /health issued 0.3s into a 2.0s /chat took 1.71s. A
    # plain `def` handler is dispatched to Starlette's threadpool, which
    # is the same fix with less machinery.
    provider = get_llm_provider()
    generation = conversation.generation()
    try:
        result = run_agent(
            user_message=req.message,
            history=conversation.snapshot(),
            provider=provider,
            tool_schemas=TOOL_SCHEMAS,
            tool_registry=TOOL_REGISTRY,
            system_prompt=BASE_SYSTEM_PROMPT,
        )
    except MaxTurnsExceeded as exc:
        # The termination guarantee firing is a designed outcome, not a
        # server fault, so this is a 200 with an honest partial answer —
        # not a 500 and a blank screen. The tool log is returned in full
        # because the work genuinely happened and is worth showing.
        #
        # History is deliberately NOT updated: the turn never reached a
        # final assistant message, and persisting a truncated transcript
        # would corrupt every subsequent turn in the conversation.
        return {
            "reply": (
                "I hit my tool-call limit before finishing this one. Here's what I did "
                "establish — but treat it as incomplete, not as a final answer. Asking a "
                "narrower question usually gets there."
            ),
            "tool_log": _tool_log_rows(exc.tool_log),
            "incomplete": True,
            "turns_used": exc.turns_used,
        }

    conversation.replace(result.messages[1:], since=generation)  # drop system msg; run_agent re-adds it
    return {
        "reply": result.final_text,
        "tool_log": _tool_log_rows(result.tool_log),
        "incomplete": False,
        "turns_used": result.turns_used,
    }


@dataclass
class _StreamEvent:
    """One item on the worker-thread -> event-source queue. `kind`
    decides how chat_stream's generator renders it; `payload` is whatever
    that kind needs (a ToolLogEntry, an AgentResult, a MaxTurnsExceeded,
    or a plain error string)."""

    kind: str  # "tool_call" | "done" | "max_turns" | "error"
    payload: Any


def _sse(event: str, data: dict) -> str:
    # SSE wire format: "event: <name>\ndata: <json>\n\n" — the blank line
    # is what marks the end of one event, not decoration.
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _run_agent_in_thread(user_message: str, history: list[dict[str, Any]], provider: Any, q: "queue.Queue[_StreamEvent]") -> None:
    """Runs the (fully synchronous, blocking-on-network) agent loop on a
    worker thread, so the asyncio event loop stays free to do anything
    else while a real LLM call is in flight. on_tool_call pushes each
    completed tool call onto `q` the instant it happens; the async
    generator in chat_stream() drains that queue concurrently and turns
    each item into an SSE event — that's the entire mechanism, no
    frameworks or extra dependencies involved."""
    try:
        result = run_agent(
            user_message=user_message,
            history=history,
            provider=provider,
            tool_schemas=TOOL_SCHEMAS,
            tool_registry=TOOL_REGISTRY,
            system_prompt=BASE_SYSTEM_PROMPT,
            on_tool_call=lambda entry: q.put(_StreamEvent("tool_call", entry)),
        )
        q.put(_StreamEvent("done", result))
    except MaxTurnsExceeded as exc:
        q.put(_StreamEvent("max_turns", exc))
    except Exception as exc:  # the worker thread has no other way to report a crash
        q.put(_StreamEvent("error", str(exc)))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    provider = get_llm_provider()
    # Snapshot the history now, not inside the worker thread: another
    # request could mutate the shared list while this one is still
    # streaming. Same single-session scope limitation as /chat — see the
    # module docstring — just made explicit here instead of implicit.
    history_snapshot = conversation.snapshot()
    # Tag the turn with the history generation it started from. If the
    # user hits Reset while this stream is still running, the generation
    # moves and the write below is dropped — otherwise /reset returns
    # {"status": "reset"}, the conversation looks cleared, and then the
    # in-flight turn silently puts the whole transcript back. One user,
    # one click, fully deterministic; not a multi-user scope issue.
    generation = conversation.generation()
    q: "queue.Queue[_StreamEvent]" = queue.Queue()
    threading.Thread(
        target=_run_agent_in_thread, args=(req.message, history_snapshot, provider, q), daemon=True
    ).start()

    async def event_source():
        while True:
            # queue.Queue.get blocks the calling thread, so it's run via
            # to_thread rather than called directly — awaiting it lets
            # the event loop serve other requests while this one waits
            # for the next tool call or the final answer.
            event: _StreamEvent = await asyncio.to_thread(q.get)
            if event.kind == "tool_call":
                entry: ToolLogEntry = event.payload
                yield _sse(
                    "tool_call",
                    {"tool_name": entry.tool_name, "arguments": entry.arguments, "result": entry.result, "error": entry.error},
                )
            elif event.kind == "done":
                result: AgentResult = event.payload
                conversation.replace(result.messages[1:], since=generation)
                yield _sse("done", {"reply": result.final_text, "incomplete": False, "turns_used": result.turns_used})
                return
            elif event.kind == "max_turns":
                exc: MaxTurnsExceeded = event.payload
                # History deliberately not updated — same reasoning as /chat.
                yield _sse(
                    "done",
                    {
                        "reply": (
                            "I hit my tool-call limit before finishing this one. Here's what I did "
                            "establish — but treat it as incomplete, not as a final answer. Asking a "
                            "narrower question usually gets there."
                        ),
                        "incomplete": True,
                        "turns_used": exc.turns_used,
                    },
                )
                return
            else:  # "error"
                yield _sse("error", {"message": str(event.payload)})
                return

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/reset")
async def reset() -> dict:
    conversation.clear()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
