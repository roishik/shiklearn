"""Unit tests for app/entity_resolution.py — the pure, zero-I/O fuzzy
matcher, now exercised against Israeli locality names in Hebrew, English,
and transliteration, as well as the original script-agnostic string
metrics. Like test_hebrew.py these must pass with no network, no API key,
and no LLM."""
from __future__ import annotations

import pytest

from app.entity_resolution import (
    DECISIVE_MIN_CONFIDENCE,
    MIN_FUZZY_QUERY_LENGTH,
    MIN_FUZZY_QUERY_LENGTH_HEBREW,
    MIN_RELEVANCE,
    jaro_similarity,
    jaro_winkler_similarity,
    resolve,
    score_pair,
    soundex,
)

# A small, realistic Israeli-locality catalog for resolve() tests: each
# entry carries its official Hebrew name, a couple of Hebrew spelling
# variants, and the English/transliterated forms a user would actually
# type -- the same SHAPE app/dataset.py's real ENTITY_CATALOG is expected
# to have (id -> every text that counts as a name for it), just built by
# hand here so these tests do not depend on the dataset module another
# agent owns.
CATALOG = {
    "tel_aviv": ["תל אביב-יפו", "Tel Aviv", "Tel Aviv-Yafo", "TLV"],
    "jerusalem": ["ירושלים", "Jerusalem", "Yerushalayim"],
    "haifa": ["חיפה", "Haifa", "Hefa"],
    "beer_sheva": ["באר שבע", "Beer Sheva", "Be'er Sheva"],
    "netanya": ["נתניה", "Netanya"],
    "raanana": ["רעננה", "Raanana", "Ra'anana"],
    "lod": ["לוד", "Lod"],
    "akko": ["עכו", "Akko", "Acre"],
}

# Deliberately NOT giving either Modiin a bare, unqualified "Modiin"
# alias, the way a real catalog builder should not either: "Modiin" on
# its own is genuinely ambiguous between these two real, different
# places, so no single alias list should claim it exclusively. That is
# what makes the ambiguity test below a real test of the RESOLVER's
# behaviour rather than an artifact of how the fixture happens to be
# built — see test_resolve_modiin_is_ambiguous_between_two_real_places.
AMBIGUOUS_CATALOG = {
    "modiin_maccabim_reut": [
        "מודיעין-מכבים-רעות",
        "Modiin-Maccabim-Reut",
        "Modi'in-Maccabim-Re'ut",
        "Modiin Maccabim Reut",
    ],
    "modiin_illit": ["מודיעין עילית", "Modiin Illit", "Modi'in Illit"],
    "ramat_gan": ["רמת גן", "Ramat Gan"],
    "ramat_hasharon": ["רמת השרון", "Ramat HaSharon"],
    "ramat_yishai": ["רמת ישי", "Ramat Yishai"],
    "kiryat_ata": ["קרית אתא", "Kiryat Ata"],
    "kiryat_bialik": ["קרית ביאליק", "Kiryat Bialik"],
    "kiryat_yam": ["קרית ים", "Kiryat Yam"],
    "kiryat_motzkin": ["קרית מוצקין", "Kiryat Motzkin"],
    "kiryat_gat": ["קרית גת", "Kiryat Gat"],
    "kfar_saba": ["כפר סבא", "Kfar Saba"],
    "kfar_yona": ["כפר יונה", "Kfar Yona"],
    "kfar_vradim": ["כפר ורדים", "Kfar Vradim"],
}

# For the short-code edit-distance fallback: kept from the seed project
# unchanged, because the fallback itself (MAX_CODE_EDIT_DISTANCE,
# _is_code_alias etc.) is generic and script-agnostic — this domain has
# no equivalent of an IATA/FAA code, so these fixtures just confirm the
# existing mechanism still works untouched by the Hebrew changes.
CODE_CATALOG = {
    "item_lgb": ["LGB", "Long Beach Airport"],
    "item_abc": ["ABC", "Alphabet Field"],
    "item_abd": ["ABD", "Alphabet Downtown"],
    "item_reno": ["Reno", "Reno Municipal Airport"],
}


# ── Jaro / Jaro-Winkler (unchanged, script-agnostic string metrics) ──────
def test_jaro_identical_strings_is_one():
    assert jaro_similarity("martha", "martha") == 1.0


def test_jaro_no_shared_characters_is_zero():
    assert jaro_similarity("abc", "xyz") == 0.0


def test_jaro_known_reference_value():
    """MARTHA/MARHTA is the canonical worked example in the literature:
    one transposition, Jaro = 0.9444."""
    assert jaro_similarity("martha", "marhta") == pytest.approx(0.9444, abs=1e-4)


def test_jaro_winkler_boosts_shared_prefix_above_plain_jaro():
    plain = jaro_similarity("martha", "marhta")
    boosted = jaro_winkler_similarity("martha", "marhta")
    assert boosted > plain
    assert boosted == pytest.approx(0.9611, abs=1e-4)


def test_jaro_empty_string_is_zero_not_a_crash():
    assert jaro_similarity("", "abc") == 0.0
    assert jaro_similarity("abc", "") == 0.0


# ── Soundex (unchanged; English-only phonetic key, see module docstring) ─
def test_soundex_known_reference_values():
    assert soundex("Robert") == "R163"
    assert soundex("Rupert") == "R163"  # classic same-code pair


def test_soundex_catches_phonetic_misspelling():
    assert soundex("Anchorage") == soundex("Ankorage")


def test_soundex_empty_input_returns_empty_not_a_crash():
    assert soundex("") == ""
    assert soundex("123") == ""


# ── score_pair: plain-Latin path, unchanged behaviour ─────────────────────
def test_score_pair_exact_match_scores_near_one():
    confidence, _ = score_pair("Tel Aviv", "Tel Aviv")
    assert confidence == pytest.approx(1.0)


def test_score_pair_is_case_and_punctuation_insensitive():
    a, _ = score_pair("tel_aviv", "TEL AVIV")
    b, _ = score_pair("tel aviv", "tel-aviv")
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(1.0)


def test_score_pair_returns_per_signal_breakdown():
    _, signals = score_pair("Haifa", "Haifa")
    assert dict(signals).keys() == {"token_containment", "jaro_winkler", "token_set", "phonetic"}


def test_score_pair_unrelated_latin_strings_score_low():
    confidence, _ = score_pair("zzzzz", "Haifa")
    assert confidence < MIN_RELEVANCE


def test_score_pair_pure_latin_path_is_unaffected_by_hebrew_support():
    """The load-bearing regression test for 'do not throw away the
    existing Latin path': when NEITHER side contains a Hebrew character,
    score_pair must take the exact fast path it always did (see its
    docstring) — this locks in specific known-good values from before
    Hebrew support existed, so a change to the Hebrew branches that
    somehow leaks into the plain-Latin path would be caught here."""
    confidence, _ = score_pair("alpha", "the alpha option")
    assert confidence == pytest.approx(0.7246, abs=1e-3)
    stopword_noise, _ = score_pair("the weather in paris", "the alpha option")
    assert confidence > stopword_noise


# ── score_pair: Hebrew-aware paths ─────────────────────────────────────────
def test_score_pair_hebrew_exact_match_scores_near_one():
    confidence, _ = score_pair("ירושלים", "ירושלים")
    assert confidence == pytest.approx(1.0)


def test_score_pair_hebrew_with_and_without_niqqud_scores_near_one():
    """'ירושלים' fully pointed vs. bare -- the same identity, and
    score_pair must recognize that through the Hebrew-normalized
    comparison branch even though the raw strings differ character-for-
    character."""
    confidence, _ = score_pair("יְרוּשָׁלַיִם", "ירושלים")
    assert confidence == pytest.approx(1.0)


def test_score_pair_hebrew_final_letter_variant_scores_near_one():
    """A name mis-spelled with a base-form letter where a sofit form is
    conventional ('ירושלימ' instead of 'ירושלים') must still be
    recognized as the same name."""
    confidence, _ = score_pair("ירושלימ", "ירושלים")
    assert confidence == pytest.approx(1.0)


def test_score_pair_hebrew_typo_scores_high_but_not_perfect():
    """A one-letter Hebrew typo (doubled yud) should still score well
    above the surfacing bar without being a literal 1.0 exact match."""
    confidence, _ = score_pair("ירושליים", "ירושלים")
    assert MIN_RELEVANCE < confidence < 1.0


def test_score_pair_cross_script_english_query_hits_hebrew_only_alias():
    """The mechanism romanize() exists for: an English query against a
    candidate string that is ONLY available in Hebrew script (no curated
    English alias present at all) must still score high enough to
    surface, via the romanized comparison branch."""
    confidence, _ = score_pair("Rehovot", "רחובות")
    assert confidence >= MIN_RELEVANCE


def test_score_pair_cross_script_hebrew_query_hits_latin_only_alias():
    """The mirror direction: a Hebrew query against a candidate string
    that is only available in Latin script."""
    confidence, _ = score_pair("רחובות", "Rehovot")
    assert confidence >= MIN_RELEVANCE


def test_score_pair_cross_script_is_symmetric_regardless_of_argument_order():
    a, _ = score_pair("Holon", "חולון")
    b, _ = score_pair("חולון", "Holon")
    assert a == pytest.approx(b)


def test_score_pair_unrelated_hebrew_strings_score_low():
    confidence, _ = score_pair("זזזזז", "ירושלים")
    assert confidence < MIN_RELEVANCE


# ── resolve(): plain-Latin behaviour, unchanged ────────────────────────────
def test_resolve_english_exact_name_is_decisive():
    result = resolve("Tel Aviv", CATALOG)
    assert result.decisive is True
    assert result.candidates[0].item_id == "tel_aviv"


def test_resolve_english_alias_is_decisive():
    result = resolve("Beer Sheva", CATALOG)
    assert result.decisive is True
    assert result.candidates[0].item_id == "beer_sheva"


def test_resolve_ranks_candidates_by_confidence_descending():
    result = resolve("Netanya", CATALOG)
    confidences = [c.confidence for c in result.candidates]
    assert confidences == sorted(confidences, reverse=True)
    assert result.candidates[0].item_id == "netanya"


def test_resolve_is_repeatable():
    r1 = resolve("Ramat", AMBIGUOUS_CATALOG)
    r2 = resolve("Ramat", AMBIGUOUS_CATALOG)
    assert [(c.item_id, c.confidence) for c in r1.candidates] == [(c.item_id, c.confidence) for c in r2.candidates]
    assert r1.decisive == r2.decisive


def test_resolve_respects_top_k():
    result = resolve("Kiryat", AMBIGUOUS_CATALOG, top_k=2)
    assert len(result.candidates) <= 2


def test_resolve_empty_catalog_returns_nothing():
    result = resolve("anything", {})
    assert result.candidates == ()
    assert result.decisive is False


def test_resolve_skips_entries_with_no_names():
    catalog = {"haifa": ["Haifa"], "broken": []}
    result = resolve("Haifa", catalog)
    assert [c.item_id for c in result.candidates] == ["haifa"]


def test_surfacing_bar_sits_below_the_acting_bar():
    assert MIN_RELEVANCE < DECISIVE_MIN_CONFIDENCE


def test_hebrew_floor_is_lower_than_latin_floor():
    """The deliberate, named tradeoff: Hebrew gets a shorter fuzzy-query
    floor than Latin because real Israeli locality names run as short as
    3 characters (see MIN_FUZZY_QUERY_LENGTH_HEBREW's own comment for the
    full justification and why it stops at 3, not lower)."""
    assert MIN_FUZZY_QUERY_LENGTH_HEBREW < MIN_FUZZY_QUERY_LENGTH


# ── resolve(): Hebrew exact / typo / niqqud / final-letter cases ──────────
def test_resolve_hebrew_exact_name_is_decisive():
    result = resolve("ירושלים", CATALOG)
    assert result.decisive is True
    assert result.candidates[0].item_id == "jerusalem"


def test_resolve_hebrew_with_typo_is_still_decisive():
    """A one-letter Hebrew typo of a real locality name, with nothing
    else in this small catalog close enough to compete, should still
    resolve decisively -- the Hebrew-script analog of the existing
    typo-tolerance tests for Latin queries."""
    result = resolve("ירושליים", CATALOG)  # doubled yud
    assert result.decisive is True
    assert result.candidates[0].item_id == "jerusalem"


def test_resolve_hebrew_without_niqqud_matches_alias_stored_with_niqqud():
    catalog = {"jerusalem": ["יְרוּשָׁלַיִם"]}  # alias stored WITH niqqud
    result = resolve("ירושלים", catalog)  # query typed WITHOUT niqqud
    assert result.decisive is True
    assert result.candidates[0].item_id == "jerusalem"


def test_resolve_hebrew_final_letter_variant_still_resolves():
    """A query mis-typed with a base-form letter where the catalog alias
    uses the conventional sofit form must still resolve to the same
    locality -- final-letter folding must survive the full resolve()
    path, not just score_pair() in isolation."""
    catalog = {"jerusalem": ["ירושלים"]}
    result = resolve("ירושלימ", catalog)  # base מ instead of sofit ם
    assert result.decisive is True
    assert result.candidates[0].item_id == "jerusalem"


def test_resolve_short_hebrew_locality_name_resolves_via_lower_floor():
    """'לוד' (Lod) is 3 characters -- exactly the length
    MIN_FUZZY_QUERY_LENGTH_HEBREW exists to admit into fuzzy scoring
    that the Latin floor (4) would have forced through exact-match-only.
    Exact match still succeeds here (it IS the correct spelling), so
    this also doubles as confirmation the lower floor doesn't somehow
    break the straightforward case."""
    result = resolve("לוד", CATALOG)
    assert result.decisive is True
    assert result.candidates[0].item_id == "lod"


# ── resolve(): English transliteration variants ────────────────────────────
@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("Jerusalem", "jerusalem"),
        ("Yerushalayim", "jerusalem"),
        ("Haifa", "haifa"),
        ("Hefa", "haifa"),
        ("Beer Sheva", "beer_sheva"),
        ("Be'er Sheva", "beer_sheva"),
        ("Netanya", "netanya"),
        ("Raanana", "raanana"),
        ("Ra'anana", "raanana"),
        ("Akko", "akko"),
        ("Acre", "akko"),
        ("TLV", "tel_aviv"),
        ("Tel Aviv-Yafo", "tel_aviv"),
    ],
)
def test_resolve_each_transliteration_variant_hits_the_right_locality(query, expected_id):
    result = resolve(query, CATALOG)
    assert result.decisive is True, f"{query!r} was not decisive: {result.candidates}"
    assert result.candidates[0].item_id == expected_id


# ── resolve(): cross-script (romanize fallback, no curated alias) ─────────
def test_resolve_cross_script_english_query_against_hebrew_only_catalog():
    """No curated English alias in the catalog at all -- this exercises
    romanize() as the ONLY path that can connect the English query to
    the Hebrew alias."""
    catalog = {"rehovot": ["רחובות"]}
    result = resolve("Rehovot", catalog)
    assert len(result.candidates) == 1
    assert result.candidates[0].item_id == "rehovot"


def test_resolve_cross_script_hebrew_query_against_latin_only_catalog():
    catalog = {"rehovot": ["Rehovot"]}
    result = resolve("רחובות", catalog)
    assert len(result.candidates) == 1
    assert result.candidates[0].item_id == "rehovot"


# ── resolve(): ambiguity MUST come back decisive=False ─────────────────────
# These are the point of the whole mechanism (per the task brief): a
# query that genuinely names more than one real place must not be
# silently resolved to whichever one happens to score a hair higher.
def test_resolve_modiin_is_ambiguous_between_two_real_places():
    """'Modiin' alone is genuinely ambiguous between Modiin-Maccabim-Reut
    and Modiin Illit -- two different, real Israeli localities that
    share a name in casual speech. Both must be surfaced; neither should
    be picked automatically."""
    result = resolve("Modiin", AMBIGUOUS_CATALOG)
    assert result.decisive is False
    assert {c.item_id for c in result.candidates} >= {"modiin_maccabim_reut", "modiin_illit"}


def test_resolve_modiin_hebrew_is_also_ambiguous():
    result = resolve("מודיעין", AMBIGUOUS_CATALOG)
    assert result.decisive is False


def test_resolve_ramat_is_ambiguous_among_three_real_places():
    """Ramat Gan, Ramat HaSharon, and Ramat Yishai are three different,
    unrelated Israeli localities that merely share the common prefix
    'Ramat' (='height of') -- a very common naming pattern, not a
    near-duplicate of ONE place."""
    result = resolve("Ramat", AMBIGUOUS_CATALOG)
    assert result.decisive is False
    assert len(result.candidates) > 1


def test_resolve_kiryat_is_ambiguous_among_many_real_places():
    """'Kiryat' (='town of') prefixes at least eight real, unrelated
    Israeli localities -- this catalog carries five of them, enough to
    prove the point without needing the full real list."""
    result = resolve("Kiryat", AMBIGUOUS_CATALOG)
    assert result.decisive is False
    assert len(result.candidates) > 1


def test_resolve_kfar_is_ambiguous_among_many_real_places():
    """'Kfar' (='village of') prefixes dozens of real Israeli
    localities."""
    result = resolve("Kfar", AMBIGUOUS_CATALOG)
    assert result.decisive is False
    assert len(result.candidates) > 1


def test_resolve_full_hyphenated_name_disambiguates_modiin_decisively():
    """The flip side of the ambiguity tests above: once the query names
    the FULL official form, it must resolve decisively — ambiguity is a
    property of the short/partial query, not of this catalog being
    unable to tell the two Modiins apart at all."""
    result = resolve("Modiin Illit", AMBIGUOUS_CATALOG)
    assert result.decisive is True
    assert result.candidates[0].item_id == "modiin_illit"


def test_resolve_full_kiryat_name_disambiguates_decisively():
    result = resolve("Kiryat Motzkin", AMBIGUOUS_CATALOG)
    assert result.decisive is True
    assert result.candidates[0].item_id == "kiryat_motzkin"


# ── resolve(): junk must return nothing, never a nearest-sounding guess ───
# The worst failure mode this component can have: silently substituting
# a plausible-looking wrong answer for a query that doesn't name anything
# in the catalog at all.
def test_resolve_latin_junk_query_returns_no_candidates():
    result = resolve("asdfgh", CATALOG)
    assert result.candidates == ()
    assert result.decisive is False


def test_resolve_hebrew_junk_query_returns_no_candidates():
    result = resolve("זזזזז", CATALOG)
    assert result.candidates == ()
    assert result.decisive is False


def test_resolve_hebrew_junk_query_returns_no_candidates_in_ambiguous_catalog():
    """Same property against the larger, prefix-heavy catalog used for
    the ambiguity tests -- junk must not accidentally brush up against
    one of the many similarly-prefixed real entries there."""
    result = resolve("קקקקקק", AMBIGUOUS_CATALOG)
    assert result.candidates == ()
    assert result.decisive is False


def test_resolve_no_match_is_not_an_exception():
    """Zero matches is a normal fuzzy-search outcome, not a caller
    mistake — deliberately unlike being handed a nonexistent item id
    elsewhere in the system, which really is a bug worth raising loudly
    for. Finding nothing for a vague or nonsense phrase is not."""
    result = resolve("quantum tunnelling diode", CATALOG)
    assert result.candidates == ()
    assert result.decisive is False


# ── short-code edit-distance fallback (unchanged; script-agnostic) ────────
def test_resolve_transposed_code_matches_when_unambiguous():
    result = resolve("LBG", CODE_CATALOG)
    assert result.decisive is True
    assert result.candidates[0].item_id == "item_lgb"


def test_resolve_code_typo_with_two_equally_close_codes_is_not_decisive():
    result = resolve("ABX", CODE_CATALOG)
    assert result.decisive is False
    assert {c.item_id for c in result.candidates} == {"item_abc", "item_abd"}


def test_resolve_short_non_code_shaped_hebrew_query_returns_nothing():
    """A 2-character Hebrew query is below even the Hebrew floor
    (MIN_FUZZY_QUERY_LENGTH_HEBREW == 3) and is not code-shaped (this
    domain has no code convention), so it must fall through to no
    candidates rather than being fuzzy-scored against every alias."""
    result = resolve("בן", CATALOG)
    assert result.candidates == ()
    assert result.decisive is False
