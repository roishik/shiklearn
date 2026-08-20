"""Tests for the pure functions in static/voice.js.

Same arrangement as tests/test_markdown_renderer.py: the shipped file is
loaded under `node` if it is installed, and skipped cleanly if not, so a
bare clone with zero setup still runs green.

Only the parts that can be tested without a microphone are here — chunk
splitting, frame energy, resampling, PCM conversion. The state machine
around them needs real audio and a real AudioContext, and a fake of both
would test the fake. That gap is stated in DESIGN_DOC.md rather than
papered over with a mock that proves nothing.

The chunk splitter earns its tests: it decides where the synthesizer is
asked to pause, and a bad split does not throw — it just makes the agent
read "zero point" and then "seven one" as two sentences. Silent
degradation is exactly what a test is for.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

VOICE_JS = Path(__file__).resolve().parent.parent / "static" / "voice.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed; voice.js tests are optional"
)


def _run(expression: str, *args: object) -> object:
    """Evaluate `expression` (with `m` bound to the module and `a` to the
    parsed arguments) under node, and return its JSON result."""
    script = (
        f"const m = require({json.dumps(str(VOICE_JS))});"
        "const a = JSON.parse(process.argv[1]);"
        f"process.stdout.write(JSON.stringify({expression}));"
    )
    proc = subprocess.run(
        ["node", "-e", script, json.dumps(list(args))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def split(text: str) -> list[str]:
    return _run("m.splitSentences(a[0])", text)  # type: ignore[return-value]


def config() -> dict:
    return _run("m.VOICE_CONFIG")  # type: ignore[return-value]


# ── VAD timing invariants ──────────────────────────────────────────────
# These are plain constants, which is exactly why they need a test: the
# bug below was two numbers drifting out of a relationship nothing
# enforced, and it produced no error — just a quietly worse transcript.


def test_preroll_covers_a_barge_in_not_just_a_normal_turn() -> None:
    """Found live in voice conversation mode: interrupting the agent ate
    the first word, while ordinary turns were fine.

    _bargeIn() seeds the utterance from the pre-roll buffer, but barge-in
    only fires after bargeInMinMs of sustained speech. If the buffer holds
    less than that, the opening has already been evicted by the time it is
    copied. At the time: preRollMs=300 < bargeInMinMs=350.

    The asymmetry is the tell — a normal turn needs only speechMinMs
    (200), which 300ms covered, so the bug hid everywhere except barge-in.
    """
    cfg = config()
    assert cfg["preRollMs"] > cfg["bargeInMinMs"], (
        f"preRollMs ({cfg['preRollMs']}) must exceed bargeInMinMs "
        f"({cfg['bargeInMinMs']}) or a barge-in loses its opening syllables"
    )


def test_preroll_carries_margin_beyond_the_bare_barge_in_minimum() -> None:
    """bargeInMinMs is a floor on SUSTAINED above-gate speech, not on
    elapsed time. During playback the gate also demands
    bargeInThresholdBoostDb, and a frame below it resets the counter, so
    real elapsed time runs longer than the minimum. Equality would be
    correct only for a speaker who starts at full volume."""
    cfg = config()
    assert cfg["preRollMs"] >= cfg["bargeInMinMs"] + 100


def test_preroll_still_covers_an_ordinary_turn() -> None:
    """The original guarantee, kept explicit so raising bargeInMinMs
    later cannot quietly cost the normal path."""
    cfg = config()
    assert cfg["preRollMs"] > cfg["speechMinMs"]


def test_a_real_utterance_is_not_discarded_as_noise() -> None:
    """minUtteranceMs drops door-slams. It must stay below the pre-roll,
    or a short but genuine barge-in ('wait') could be thrown away as
    noise before it is ever transcribed."""
    cfg = config()
    assert cfg["minUtteranceMs"] < cfg["preRollMs"]


# ── Chunk splitting ────────────────────────────────────────────────────


def test_a_decimal_is_never_a_sentence_boundary() -> None:
    # The failure this file exists for. Every reply in this domain is full
    # of scores like 0.71, and splitting there makes the agent pause
    # mid-number.
    chunks = split("Ra'anana scores 0.71 and Kfar Saba scores 0.66 on the same index.")
    assert len(chunks) == 1


def test_real_sentences_do_split() -> None:
    long_first = "Ra'anana leads the northern locality shortlist on the value-for-money score."
    long_second = "Kfar Saba follows it closely, and Hadera is a distant third."
    chunks = split(f"{long_first} {long_second}")
    assert chunks == [long_first, long_second]


def test_short_fragments_are_merged_forward() -> None:
    # "Yes." is not worth a network round-trip, and neither is a short
    # follow-on fragment.
    chunks = split("Yes. Prices vary. The strongest value pick up north is Ra'anana.")
    assert len(chunks) == 1
    assert chunks[0].startswith("Yes. Prices vary.")


def test_the_first_chunk_stays_small_enough_to_start_quickly() -> None:
    # Time-to-first-audio is set by the first chunk. If the splitter ever
    # starts emitting the whole reply as chunk one, the conversation goes
    # quiet for seconds before it speaks.
    reply = " ".join(
        [
            "Ra'anana leads the northern shortlist with a value-for-money score of 0.71.",
            "Kfar Saba follows at 0.66, and Hadera trails at 0.61.",
            "The gap between the top two is inside the weighting judgement that produced it.",
            "Ra'anana scores highest on price growth and on population growth.",
        ]
    )
    chunks = split(reply)
    assert len(chunks) > 1
    assert len(chunks[0]) <= config()["maxSpeakChars"]


def test_an_overlong_sentence_is_broken_at_a_clause_boundary() -> None:
    clause = "which is a long clause about the local amenities and the price trend"
    sentence = "Ra'anana leads, " + ", ".join([clause] * 6) + "."
    chunks = split(sentence)
    assert len(chunks) > 1
    assert all(len(c) <= config()["maxSpeakChars"] + 1 for c in chunks)
    # Rejoining must not lose text — a dropped clause is an agent that
    # says something different from what it wrote.
    assert "".join(chunks).replace(" ", "") == sentence.replace(" ", "")


def test_empty_and_whitespace_input_produce_no_chunks() -> None:
    assert split("") == []
    assert split("   \n\t ") == []


def test_question_and_exclamation_marks_are_boundaries() -> None:
    chunks = split(
        "Which locality did you mean by Haifa, since it has several distinct neighborhoods? "
        "I will assume the city center unless you say otherwise."
    )
    assert len(chunks) == 2


# ── Signal helpers ─────────────────────────────────────────────────────


def test_silence_reads_as_the_noise_floor() -> None:
    assert _run("m.dbfsOfFrame(new Float32Array(256))") == -120


def test_full_scale_reads_as_zero_dbfs() -> None:
    value = _run("m.dbfsOfFrame(Float32Array.from({length: 256}, () => 1))")
    assert abs(float(value)) < 0.001  # type: ignore[arg-type]


def test_speech_level_audio_clears_the_gate_and_room_tone_does_not() -> None:
    gate = config()["speechThresholdDbfs"]
    speech = _run("m.dbfsOfFrame(Float32Array.from({length: 512}, (_, i) => 0.1 * Math.sin(i / 4)))")
    quiet = _run("m.dbfsOfFrame(Float32Array.from({length: 512}, (_, i) => 0.0005 * Math.sin(i / 4)))")
    assert float(speech) >= gate  # type: ignore[arg-type]
    assert float(quiet) < gate  # type: ignore[arg-type]


def test_downsampling_produces_the_expected_frame_length() -> None:
    # 48 kHz in, 16 kHz out: exactly one third of the samples.
    assert _run("m.downsample(new Float32Array(480), 48000, 16000).length") == 160


def test_downsampling_is_a_no_op_at_the_target_rate() -> None:
    assert _run("Array.from(m.downsample(Float32Array.from([0.1, -0.2]), 16000, 16000))") == [
        pytest.approx(0.1, abs=1e-6),
        pytest.approx(-0.2, abs=1e-6),
    ]


def test_pcm_conversion_clamps_at_the_rails() -> None:
    # An over-unity float wrapping around instead of clamping is a loud
    # click in the transcriber's input.
    assert _run("Array.from(m.floatToPcm16(Float32Array.from([0, 1, -1, 2, -2])))") == [
        0,
        32767,
        -32768,
        32767,
        -32768,
    ]


# ── Tuning invariants ──────────────────────────────────────────────────


def test_barge_in_is_harder_to_trigger_than_a_normal_turn_start() -> None:
    # Deliberate asymmetry: a false barge-in cuts the agent off mid-answer,
    # which is worse than a slightly late one. If these ever equalize, the
    # asymmetry was lost in a refactor.
    cfg = config()
    assert cfg["bargeInMinMs"] > cfg["speechMinMs"]
    assert cfg["bargeInThresholdBoostDb"] > 0


def test_pre_roll_exists_so_the_first_syllable_survives() -> None:
    # The gate only opens after speech has already started, so without
    # pre-roll every transcript loses its opening sound.
    assert config()["preRollMs"] > 0


def test_the_utterance_bounds_are_sane() -> None:
    cfg = config()
    assert 0 < cfg["minUtteranceMs"] < cfg["maxUtteranceMs"]
    assert cfg["minSpeakChars"] < cfg["maxSpeakChars"]
    assert cfg["synthesisLookahead"] >= 1
