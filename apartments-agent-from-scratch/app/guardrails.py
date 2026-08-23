"""
guardrails.py — tool output, and any other retrieved/external text, is
UNTRUSTED DATA, never instructions.

Any agent that feeds fetched data back into its own prompt inherits this
problem: text arriving from an API, a scraped page, or a pasted document
can carry "ignore previous instructions" / "reveal your system prompt"
payloads that read to the model as if they came from the operator.

This module gives agent_loop.py two things:

  1. wrap_untrusted(text, source) — fences arbitrary text so it's
     unambiguous, IN THE PROMPT ITSELF, that it's data to read, not
     directions to follow. Every tool result gets wrapped before it's
     appended to the message history (see agent_loop.py).
  2. scan_for_injection(text) — a cheap, deterministic pre-filter that
     flags common prompt-injection phrasings BEFORE the text reaches the
     model, so a suspicious tool result can be logged / surfaced / blocked
     rather than silently trusted.

This is not a substitute for a real classifier, an allow-listed tool
surface, or output-side data-loss checks in production — it is the
minimum "treat tool output as data" discipline, made visible, logged, and
testable. See README 'Guardrails — what this is and isn't'.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

UNTRUSTED_CLOSE = "</untrusted_data>"

# Deliberately simple regexes with an obvious name, so this list is easy to
# read and easy to extend. Not exhaustive — see README for
# what a production version would add (semantic classifier, allow-listed
# tool schemas, output scanning for exfiltrated secrets, etc).
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore (all )?(the )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (all )?(the )?(previous|prior|above)", re.I),
    re.compile(r"reveal (your |the )?system prompt", re.I),
    re.compile(r"print (your |the )?system prompt", re.I),
    re.compile(r"show (me )?(your |the )?(system prompt|instructions)", re.I),
    re.compile(r"you are now (in )?(a )?(developer|debug|dan) mode", re.I),
    re.compile(r"act as (if you (are|were)|an?) (an? )?(unfiltered|unrestricted|jailbroken)", re.I),
    re.compile(r"new instructions?\s*:", re.I),
    re.compile(r"</?(system|assistant)[ >]", re.I),  # fake role-tag injection
    # This module's OWN fence. Fake role tags were anticipated above and
    # the tag we actually rely on was the one missed — an untrusted string
    # containing our closing delimiter is trying to end the quarantine
    # early and be read as developer text.
    re.compile(r"</?untrusted_data", re.I),
    re.compile(r"do anything now", re.I),
    re.compile(r"pretend (you are|to be) (an? )?(ai )?(with no|without) (restrictions|rules|guardrails)", re.I),
    # Hebrew equivalents of the highest-value patterns above. This project
    # treats Hebrew as a first-class input language throughout (see
    # app/hebrew.py, the RTL UI work) -- an injection scanner that only
    # recognizes these phrasings in English would have ZERO coverage for
    # exactly the language this agent is built to accept, which is a real
    # gap, not the same kind of "misses clever paraphrasing" limitation
    # the module docstring already discloses. Literal common phrasings
    # only, same scope and same honesty about that scope as the English
    # list above -- Hebrew's prefix morphology (ה-/ל-/מ-/ו-/ש-/כ- attach
    # directly to the next word) and verb-conjugation-by-gender make an
    # exhaustive regex impractical here just as it is in English; this
    # catches what someone would actually type, not everything they could.
    re.compile(r"התעלם מ.{0,15}הוראות ה?קודמות", re.I),  # "ignore the previous instructions"
    # {0,20} rather than {0,10}: "the system's prompt" is naturally phrased
    # "...של המערכת" (prompt OF the system) in Hebrew, not as a tight
    # compound the way English says "system prompt" -- a narrower gap
    # missed this real, natural phrasing in testing.
    re.compile(r"חשוף.{0,20}(פרומפט|הנחיות).{0,10}מערכת", re.I),  # "reveal the system's prompt/instructions"
    re.compile(r"הצג.{0,20}(פרומפט|הנחיות).{0,10}מערכת", re.I),  # "show/print the system's prompt"
    re.compile(r"את[ה]? (כעת|עכשיו) במצב (מפתח|דיבוג|מפתחים)", re.I),  # "you are now in developer/debug mode"
    re.compile(r"הוראות חדשות\s*:", re.I),  # "new instructions:"
    re.compile(r"התחזה ל.{0,20}(ללא|בלי) (הגבלות|כללים)", re.I),  # "pretend to be ... without restrictions/rules"
]


@dataclass(frozen=True)
class ScanResult:
    flagged: bool
    matched_patterns: tuple[str, ...]


def scan_for_injection(text: str) -> ScanResult:
    """Deterministic, regex-based scan for common prompt-injection
    phrasings. Pure function — no I/O, safe to unit test in isolation."""
    if not text:
        return ScanResult(flagged=False, matched_patterns=())
    matches = tuple(p.pattern for p in _INJECTION_PATTERNS if p.search(text))
    return ScanResult(flagged=bool(matches), matched_patterns=matches)


def wrap_untrusted(text: str, source: str) -> str:
    """Fence external/tool text so the model can't mistake it for
    developer/system instructions, and annotate it if the deterministic
    scanner flagged a likely injection attempt. ALWAYS call this on tool
    output before appending it to the conversation — see agent_loop.py,
    which does this for every tool result with no exceptions.

    The flagged text is still INCLUDED (not silently dropped) — visibility
    beats silent filtering for a take-home demo where "show the guardrail
    catching it" is the point. A production system would additionally
    decide whether to block, redact, or route flagged content to a human;
    see README."""
    scan = scan_for_injection(text)
    header = f'<untrusted_data source="{source}">'
    if scan.flagged:
        header += f"\n<!-- guardrail: flagged patterns {list(scan.matched_patterns)} -->"
    # Neutralize any copy of the fence inside the payload BEFORE fencing.
    # Otherwise the wrapper's own closing tag can appear in the middle of
    # the quarantined text, and everything after it reads as if the
    # quarantine had ended. The source is not hypothetical: locality and
    # neighborhood names come from a public, community-editable dataset,
    # and tool results can put them verbatim into a response field.
    #
    # A zero-width space inside the tag, rather than deleting the text:
    # the reader still sees exactly what the data said, which is the same
    # visibility-over-filtering principle as flagging rather than dropping.
    # Scanning happens first, so the attempt is still reported.
    text = _defuse_fence(text)
    return f"{header}\n{text}\n{UNTRUSTED_CLOSE}"


def _defuse_fence(text: str) -> str:
    """Break any literal `untrusted_data` tag in a payload so it cannot
    terminate the fence that is about to be wrapped around it."""
    return re.sub(r"<(/?)(untrusted_data)", "<\u200b\\1\\2", text, flags=re.I)


def unwrap_untrusted(wrapped: str) -> str:
    """Inverse of wrap_untrusted: strip the fence and any guardrail
    comment line, returning the original payload. The real LLM providers
    never need this (they just read the fenced text as part of the
    conversation) — it exists for callers that need the raw payload back,
    e.g. app/providers/llm/mock_llm.py parsing a tool's JSON result, or a
    log viewer that wants to show the clean payload separately from the
    guardrail annotation."""
    lines = wrapped.splitlines()
    if lines and lines[0].startswith("<untrusted_data"):
        lines = lines[1:]
    if lines and lines[0].startswith("<!-- guardrail:"):
        lines = lines[1:]
    if lines and lines[-1].strip() == UNTRUSTED_CLOSE:
        lines = lines[:-1]
    return "\n".join(lines)
