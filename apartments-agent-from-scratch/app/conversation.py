"""The in-memory conversation history, and the one operation voice adds
to it: truncating a reply the user talked over.

This lives in its own module because two request paths now share it —
`app/main.py` (text chat) and `app/voice_api.py` (spoken chat) — and
having the voice endpoints reach into the web module for a private list
would have made the import graph circular for no benefit.

SCOPE, stated plainly: this is a single process-wide history, which is
right for a single-user demo and wrong for anything multi-user. Two
browser tabs share one conversation. Making it per-session is a dict
keyed by a session cookie and about fifteen lines, deliberately not
spent here — see README "Scope and limitations".
"""
from __future__ import annotations

from typing import Any

# Message list in the OpenAI chat format, WITHOUT the system message —
# `agent_loop.run_agent` prepends that itself on every call, so storing it
# here would double it.
_history: list[dict[str, Any]] = []

# Bumped on every write and every clear. A turn takes seconds and the user
# can press Reset in the middle of one, so a write has to be able to ask
# "is the conversation I started from still the current one?" — see
# `replace(since=...)`.
_generation = 0


def generation() -> int:
    """The current history generation. Callers snapshot this alongside
    the history and hand it back when they write."""
    return _generation


def snapshot() -> list[dict[str, Any]]:
    """A copy, not the live list. A streaming request must not observe
    another request mutating history halfway through its own turn."""
    return list(_history)


def replace(messages: list[dict[str, Any]], *, since: int | None = None) -> bool:
    """Store a completed turn, unless the conversation moved underneath it.

    `since` is the generation the caller snapshotted before starting. If
    it no longer matches, the user reset (or another turn landed) while
    this one was in flight, and writing now would resurrect a
    conversation they were told was gone. Returns whether the write
    happened.
    """
    global _generation
    if since is not None and since != _generation:
        return False
    _history[:] = messages
    _generation += 1
    return True


def clear() -> None:
    global _generation
    _history.clear()
    _generation += 1


def truncate_last_reply(spoken_prefix: str) -> bool:
    """Rewrite the most recent assistant reply to only what the user
    actually heard, and report whether anything was rewritten.

    This is the third step of barge-in, and the one that is easy to skip
    and expensive to get wrong. Steps one and two — stop speaking, drop
    the audio already queued — are what the user perceives. This step is
    what keeps the conversation coherent afterwards: if the agent was cut
    off four sentences into a nine-sentence answer, the model must not
    believe it said the other five. Otherwise the next turn is built on a
    transcript that never happened, and follow-ups like "what was the
    third one?" answer from text the user never heard.

    Granularity is whole sentences, because sentences are the unit the
    client synthesizes and plays. Word-level truncation would need
    per-word timing from the TTS provider, which neither provider here
    returns from its plain synthesis endpoint. Named limitation, not an
    oversight — see DECISIONS.md.

    An empty `spoken_prefix` means the user interrupted before any audio
    played at all, so the reply is removed outright rather than stored as
    an empty assistant turn.

    THE PREFIX MUST ACTUALLY BE A PREFIX, and that check is a correctness
    boundary rather than paranoia. Without it this function will write
    whatever string it is handed into the assistant's own turn, and the
    model then treats it as something it previously said. That is the
    cheapest possible defeat of this project's central claim — one HTTP
    request and the agent "remembers" stating a score no code computed.
    Barge-in only ever needs to REMOVE text, so refusing anything that is
    not a leading substring of the stored reply costs nothing and makes
    the operation structurally incapable of adding a fact.
    """
    # Only the MOST RECENT assistant turn is a candidate. Scanning further
    # back would mean that a turn which ended with an empty reply causes
    # the PREVIOUS answer to be truncated instead — rewriting a turn the
    # user is not talking over.
    for i in range(len(_history) - 1, -1, -1):
        message = _history[i]
        if message.get("role") != "assistant":
            continue
        if not message.get("content"):
            # The newest assistant message is a tool-call turn or an empty
            # reply: there is nothing the user could have heard. Stop
            # rather than reaching past it.
            return False
        stored = str(message["content"])
        prefix = spoken_prefix.strip()
        if prefix and not _is_spoken_prefix_of(prefix, stored):
            return False
        if prefix:
            _history[i] = {**message, "content": prefix}
        else:
            del _history[i]
        return True
    return False


def _is_spoken_prefix_of(prefix: str, stored: str) -> bool:
    """Whether `prefix` is the opening of `stored`, ignoring whitespace.

    Compared with whitespace collapsed because the client sends back the
    chunks it actually played, which have been through markdown stripping
    and re-joining — so the spacing legitimately differs from the stored
    markdown even when the words are identical. Everything else must
    match exactly, and it must match from the start.
    """
    normalize = lambda text: " ".join(text.split())
    return normalize(stored).startswith(normalize(prefix))
