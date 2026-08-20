"""Tests for app/conversation.py — the shared history, and barge-in's
third step.

`truncate_last_reply` is the one piece of the voice feature whose failure
is completely invisible at the time it happens. Steps one and two of a
barge-in are obvious if they break: audio keeps playing. This one only
shows up a turn later, when the agent answers a follow-up using text the
user never heard. That is exactly the kind of bug a test is for.
"""
from __future__ import annotations

import pytest

from app import conversation


@pytest.fixture(autouse=True)
def clean_history():
    conversation.clear()
    yield
    conversation.clear()


def _seed(*messages: dict) -> None:
    conversation.replace(list(messages))


def test_snapshot_is_a_copy_not_the_live_list() -> None:
    # A streaming request holds its snapshot for seconds while another
    # request may replace the history underneath it. If snapshot() handed
    # out the real list, the in-flight turn would see the mutation.
    _seed({"role": "user", "content": "hello"})
    snap = conversation.snapshot()
    snap.append({"role": "user", "content": "injected"})
    assert len(conversation.snapshot()) == 1


def test_truncate_rewrites_the_last_assistant_reply() -> None:
    _seed(
        {"role": "user", "content": "rank the north"},
        {"role": "assistant", "content": "Ra'anana leads. Kfar Saba follows. Hadera is third."},
    )
    assert conversation.truncate_last_reply("Ra'anana leads.") is True
    assert conversation.snapshot()[-1] == {"role": "assistant", "content": "Ra'anana leads."}


def test_truncate_leaves_earlier_turns_untouched() -> None:
    _seed(
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second answer, long version"},
    )
    conversation.truncate_last_reply("second answer")
    history = conversation.snapshot()
    assert history[1]["content"] == "first answer"
    assert history[3]["content"] == "second answer"


def test_truncate_with_an_empty_prefix_removes_the_reply_entirely() -> None:
    # Interrupted before a single word played. Storing an empty assistant
    # turn would leave the model believing it answered and said nothing.
    _seed(
        {"role": "user", "content": "rank the north"},
        {"role": "assistant", "content": "Ra'anana leads. Kfar Saba follows."},
    )
    assert conversation.truncate_last_reply("") is True
    assert conversation.snapshot() == [{"role": "user", "content": "rank the north"}]


def test_truncate_skips_tool_call_messages_to_find_the_reply() -> None:
    # A real turn ends with tool messages and an assistant message that
    # has tool_calls but no content. Neither of those is what the user
    # heard; the last assistant message WITH content is.
    _seed(
        {"role": "user", "content": "compare 5000 and 3000"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
        {"role": "assistant", "content": "5000 ranks first. 3000 is second."},
    )
    assert conversation.truncate_last_reply("5000 ranks first.") is True
    history = conversation.snapshot()
    assert history[-1]["content"] == "5000 ranks first."
    assert history[1]["tool_calls"] == [{"id": "1"}]  # untouched


def test_truncate_preserves_other_keys_on_the_message() -> None:
    _seed({"role": "assistant", "content": "long answer", "name": "agent"})
    conversation.truncate_last_reply("long")
    assert conversation.snapshot()[-1] == {"role": "assistant", "content": "long", "name": "agent"}


def test_truncate_reports_false_when_there_is_nothing_to_truncate() -> None:
    # An interrupt can legitimately arrive with no reply to cut — the user
    # spoke over the very first "thinking" pause. That is not an error,
    # and it must not throw.
    assert conversation.truncate_last_reply("anything") is False
    _seed({"role": "user", "content": "only a question"})
    assert conversation.truncate_last_reply("anything") is False


def test_whitespace_only_prefix_is_treated_as_nothing_heard() -> None:
    _seed({"role": "assistant", "content": "Ra'anana leads."})
    conversation.truncate_last_reply("   \n  ")
    assert conversation.snapshot() == []


def test_clear_empties_everything() -> None:
    _seed({"role": "user", "content": "x"}, {"role": "assistant", "content": "y"})
    conversation.clear()
    assert conversation.snapshot() == []
