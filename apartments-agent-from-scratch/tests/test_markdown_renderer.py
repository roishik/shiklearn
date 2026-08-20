"""Tests for static/markdown.js — the agent's replies, rendered.

Why a JavaScript file is tested from a Python suite: the renderer is the
only place in this repo where model output is turned into live HTML, so
its escape-first invariant is worth an assertion rather than a comment.
Adding a JS toolchain (npm, a bundler, a test runner) to a four-package
Python project to test ~90 lines would cost more than it returns, so the
tests shell out to `node` if it happens to be installed and skip cleanly
if it is not. The skip is deliberate and safe: nothing else in the app
depends on node, and `pytest` must stay green on a bare clone with zero
setup (see app/config.py's docstring for the same principle applied to
API keys).

The file under test is loaded directly, not copied — a duplicated
fixture would drift away from what actually ships.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

MARKDOWN_JS = Path(__file__).resolve().parent.parent / "static" / "markdown.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed; markdown.js tests are optional"
)


def _call(fn: str, source: str) -> str:
    """Run one exported function from static/markdown.js under node."""
    script = (
        f"const m = require({json.dumps(str(MARKDOWN_JS))});"
        f"process.stdout.write(m[{json.dumps(fn)}](JSON.parse(process.argv[1])));"
    )
    proc = subprocess.run(
        ["node", "-e", script, json.dumps(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def render(source: str) -> str:
    return _call("renderMarkdown", source)


def speak(source: str) -> str:
    return _call("stripMarkdown", source)


# ── The safety property ────────────────────────────────────────────────
# These are the tests that justify the file existing. Everything below
# them is behaviour; these are the invariant.


# Every tag the renderer is allowed to emit. Anything outside this set
# appearing in output means source text became live markup.
ALLOWED_TAGS = {
    "p", "br", "strong", "em", "del", "code", "pre", "hr", "a",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote",
    "table", "thead", "tbody", "tr", "th", "td",
}


def _tags(html: str) -> set[str]:
    return {m.lower() for m in re.findall(r"</?([a-zA-Z][a-zA-Z0-9]*)", html)}


def test_html_in_source_is_never_live_markup() -> None:
    out = render("A <script>alert(1)</script> and <img src=x onerror=alert(1)>")
    # The escaped text is still THERE — the renderer does not censor the
    # reply, it neutralizes it. What matters is that no tag survived.
    assert "&lt;script&gt;" in out
    assert "&lt;img src=x onerror=alert(1)&gt;" in out
    assert _tags(out) <= ALLOWED_TAGS


def test_no_source_can_introduce_a_tag_the_renderer_does_not_emit() -> None:
    hostile = (
        "<iframe src=evil></iframe>\n\n"
        "- <svg/onload=alert(1)>\n"
        "- **<style>body{}</style>**\n\n"
        "| <form> | <input> |\n|---|---|\n| <object> | <embed> |\n\n"
        "> <base href=x>\n\n"
        "`<link rel=x>`\n"
    )
    assert _tags(render(hostile)) <= ALLOWED_TAGS


def test_html_inside_a_code_span_is_still_escaped() -> None:
    # Code spans are lifted out into placeholders before the emphasis
    # passes run; this checks that lifting them out does not smuggle them
    # past the escape, which happens earlier and only once.
    out = render("Call `<script>` carefully")
    assert "<script>" not in out
    assert "<code>&lt;script&gt;</code>" in out


def test_link_scheme_is_allowlisted() -> None:
    out = render("[safe](https://www.cbs.gov.il) [bad](javascript:alert(1)) [also-bad](data:text/html,x)")
    assert '<a href="https://www.cbs.gov.il"' in out
    assert out.count("<a ") == 1  # only the https link became a link
    assert 'href="javascript:' not in out
    assert 'href="data:' not in out
    # The rejected ones stay visible as literal markdown rather than
    # vanishing — a link the UI refuses to make clickable is still
    # something the reader should be able to see.
    assert "[bad](javascript:alert(1))" in out


def test_link_gets_noopener() -> None:
    out = render("[docs](https://www.cbs.gov.il/localities)")
    assert 'rel="noopener noreferrer"' in out


# ── Block-level rendering ──────────────────────────────────────────────


def test_headings_and_emphasis() -> None:
    # No apostrophe in this name on purpose: escapeHtml() turns ' into
    # &#39;, and a literal-string assertion on the rendered output would
    # otherwise be asserting on the escaped entity rather than the tag.
    out = render("## Ranking\n\n**Netanya** ranks first, *narrowly*.")
    assert "<h2>Ranking</h2>" in out
    assert "<strong>Netanya</strong>" in out
    assert "<em>narrowly</em>" in out


def test_ordered_list_with_nested_bullets() -> None:
    out = render("1. Ra'anana\n2. Haifa\n   - growth: 0.9\n   - size: 0.2\n3. Ashdod")
    assert out.startswith("<ol>")
    assert "<ul><li>growth: 0.9</li><li>size: 0.2</li></ul></li>" in out
    assert out.count("<li>") == 5


def test_table_renders_as_a_table() -> None:
    out = render("| Locality | Score |\n|---|---:|\n| Ra'anana | 0.71 |\n| Haifa | 0.69 |")
    assert "<th>Locality</th>" in out
    assert "<td>0.71</td>" in out
    assert out.count("<tr>") == 3  # header + two body rows


def test_fenced_code_block() -> None:
    out = render('```json\n{"score": 0.71}\n```')
    assert "<pre><code>" in out
    assert "&quot;score&quot;" in out


def test_blockquote() -> None:
    # '>' is an entity by the time the block parser sees it — this is the
    # test that catches the renderer matching the raw character instead,
    # which fails silently rather than loudly.
    out = render("&gt; not a quote\n\n> a real quote")
    assert "<blockquote>a real quote</blockquote>" in out


def test_plain_paragraph_keeps_single_newlines_as_breaks() -> None:
    assert render("Line one\nLine two") == "<p>Line one<br>Line two</p>"


def test_empty_source_renders_nothing() -> None:
    assert render("") == ""


# ── The speech path ────────────────────────────────────────────────────


def test_speech_strips_emphasis_and_headers() -> None:
    assert speak("## Ranking\n\n**Ra'anana** ranks first.") == "Ranking\n\nRa'anana ranks first."


def test_speech_turns_table_pipes_into_pauses_without_leading_commas() -> None:
    spoken = speak("| Locality | Score |\n|---|---:|\n| Ra'anana | 0.71 |")
    assert not any(line.strip().startswith(",") for line in spoken.splitlines())
    assert "Locality, Score" in spoken


def test_speech_drops_fenced_code() -> None:
    # Reading raw tool JSON aloud is worse than saying nothing about it.
    assert "score" not in speak('Here it is:\n\n```json\n{"score": 0.71}\n```')


def test_speech_keeps_link_text_and_drops_the_url() -> None:
    assert speak("See [the CBS page](https://www.cbs.gov.il/x) for detail") == (
        "See the CBS page for detail"
    )
