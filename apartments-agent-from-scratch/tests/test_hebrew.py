"""Unit tests for app/hebrew.py — the pure, zero-I/O Hebrew text layer.
Like test_entity_resolution.py, these must pass with no network, no API
key, and no LLM."""
from __future__ import annotations

from app.hebrew import (
    REGION_GROUPS,
    REGION_NAME_ALIASES,
    TRANSLITERATION_VARIANTS,
    is_hebrew,
    match_region,
    normalize_hebrew,
    romanize,
    strip_niqqud,
)


# ── strip_niqqud ────────────────────────────────────────────────────────
def test_strip_niqqud_removes_points():
    """יְרוּשָׁלַיִם is 'ירושלים' (Jerusalem) fully pointed. Stripping
    niqqud must reduce it back to the bare consonantal spelling."""
    assert strip_niqqud("יְרוּשָׁלַיִם") == "ירושלים"


def test_strip_niqqud_no_op_on_unpointed_text():
    assert strip_niqqud("ירושלים") == "ירושלים"


def test_strip_niqqud_no_op_on_latin_text():
    assert strip_niqqud("Jerusalem") == "Jerusalem"


def test_strip_niqqud_empty_string():
    assert strip_niqqud("") == ""


# ── is_hebrew ───────────────────────────────────────────────────────────
def test_is_hebrew_true_for_hebrew_text():
    assert is_hebrew("ירושלים") is True


def test_is_hebrew_false_for_latin_text():
    assert is_hebrew("Jerusalem") is False


def test_is_hebrew_false_for_digits_and_punctuation():
    assert is_hebrew("12345!?") is False


def test_is_hebrew_true_for_mixed_script():
    """A single Hebrew character anywhere in the string is enough --
    this is a coarse gate used upstream to decide whether the Hebrew
    branches of entity_resolution are worth running at all, not a
    strict 'is this string ENTIRELY Hebrew' check."""
    assert is_hebrew("Kfar כפר") is True


def test_is_hebrew_false_for_empty_string():
    assert is_hebrew("") is False


# ── normalize_hebrew ────────────────────────────────────────────────────
def test_normalize_hebrew_strips_niqqud():
    assert normalize_hebrew("יְרוּשָׁלַיִם") == normalize_hebrew("ירושלים")


def test_normalize_hebrew_folds_final_letters():
    """A name spelled with a mid-word (base) form where a final form is
    conventional, and the same name spelled correctly, must collate --
    that's the entire point of the fold. 'ירושלים' ends in sofit-mem
    (ם); the fold reduces it to base-mem (מ), so a deliberately
    mis-spelled 'ירושלימ' (base mem instead of sofit) collates with the
    correctly-spelled form once both are folded."""
    assert normalize_hebrew("ירושלים") == normalize_hebrew("ירושלימ")


def test_normalize_hebrew_folds_each_final_letter():
    # ם/מ, ן/נ, ץ/צ, ף/פ, ך/כ -- one worked example per pair, using
    # minimal made-up strings so each fold is isolated and unambiguous.
    assert normalize_hebrew("שלום") == normalize_hebrew("שלומ")  # ם -> מ
    assert normalize_hebrew("דן") == normalize_hebrew("דנ")  # ן -> נ
    assert normalize_hebrew("ארץ") == normalize_hebrew("ארצ")  # ץ -> צ
    assert normalize_hebrew("כף") == normalize_hebrew("כפ")  # ף -> פ
    assert normalize_hebrew("ברך") == normalize_hebrew("ברכ")  # ך -> כ


def test_normalize_hebrew_folds_geresh_and_gershayim_to_ascii():
    assert normalize_hebrew("ת׳") == "ת'"  # geresh -> ASCII apostrophe
    assert normalize_hebrew("ת״א") == 'ת"א'  # gershayim -> ASCII quote


def test_normalize_hebrew_folds_maqaf_to_ascii_hyphen():
    assert normalize_hebrew("מודיעין־מכבים") == "מודיעינ-מכבימ"  # note: sofit ן/ם also fold, see the dedicated final-letter tests


def test_normalize_hebrew_maqaf_survives_alongside_niqqud_stripping():
    """Regression for an order-of-operations bug found while building
    this module: maqaf (U+05BE) is numerically INSIDE strip_niqqud's
    U+0591-U+05C7 range, so stripping niqqud before folding maqaf would
    silently delete the word boundary instead of turning it into a
    hyphen. This pins the correct order (fold maqaf first)."""
    result = normalize_hebrew("מודיעין־מכבים־רעות")
    assert result == "מודיעינ-מכבימ-רעות"  # sofit letters fold too; see the maqaf itself is what this test pins
    assert "־" not in result


def test_normalize_hebrew_collapses_whitespace_runs():
    assert normalize_hebrew("תל   אביב") == "תל אביב"


def test_normalize_hebrew_collapses_hyphen_runs():
    assert normalize_hebrew("מודיעין--מכבים") == "מודיעינ-מכבימ"  # sofit letters fold too; hyphen-collapse is what this test pins


def test_normalize_hebrew_hyphens_stay_distinct_from_spaces():
    """Collapsing hyphens to SPACES (rather than collapsing hyphen runs
    to a single hyphen) would erase the fact that 'מודיעין-מכבים-רעות'
    is conventionally written as one hyphenated compound -- romanize()
    depends on the hyphen surviving to treat each part positionally.
    This is a regression test for exactly that bug."""
    assert normalize_hebrew("תל אביב-יפו") == "תל אביב-יפו"
    assert "-" in normalize_hebrew("תל אביב-יפו")


def test_normalize_hebrew_trims_boundary_hyphen_from_maqaf():
    """A maqaf at a word boundary (e.g. from a preceding definite
    article that was itself dropped) must not leave a stray leading or
    trailing hyphen behind."""
    assert normalize_hebrew("־מודיעין־") == "מודיעינ"  # sofit ן folds to נ; boundary hyphens are what this test pins


def test_normalize_hebrew_no_op_on_plain_latin_text():
    """A non-Hebrew string passed to this Hebrew-specific function
    should not be mangled -- it should come back with, at most, its
    whitespace collapsed, exactly like entity_resolution's own
    _normalize would leave it. This matters because score_pair() in
    entity_resolution.py calls normalize_hebrew() unconditionally on
    BOTH sides whenever either side has Hebrew, so it must be a safe
    no-op on the other, non-Hebrew side."""
    assert normalize_hebrew("Tel Aviv") == "Tel Aviv"


def test_normalize_hebrew_empty_string():
    assert normalize_hebrew("") == ""


def test_normalize_hebrew_is_pure_and_repeatable():
    text = "יְרוּשָׁלַיִם־ test"
    assert normalize_hebrew(text) == normalize_hebrew(text)


# ── romanize ────────────────────────────────────────────────────────────
def test_romanize_hand_verified_simple_names():
    """Hand-verified against this module's own documented letter rules
    (see romanize()/_romanize_word's docstrings), not against an
    external transliteration standard -- the point of this function is
    a documented, deterministic mapping, and these pin exactly what
    that mapping currently produces so a future change to the letter
    table is a visible, deliberate diff here."""
    assert romanize("לוד") == "lod"  # ל=l ו=o(non-initial) ד=d
    assert romanize("דן") == "dn"  # ד=d, sofit ן folds to נ=n (both consonants, no written vowel)


def test_romanize_word_initial_bet_is_b_mid_word_is_v():
    """'ב' carries the word-initial-dagesh heuristic documented on
    _romanize_word: hard 'b' at the start of a word, soft 'v' elsewhere.
    'אביב' (Aviv) isolates this: א is silent, leaving ב as the very
    first rendered letter -- but it is NOT the first LETTER of the word,
    so it still gets the mid-word 'v' reading."""
    assert romanize("אביב") == "viv"


def test_romanize_word_initial_pe_is_p_mid_word_is_f():
    """Same heuristic, applied to פ -- 'חיפה' (Haifa) needs the mid-word
    reading to land as 'f' for the romanized form to resemble the
    common English spelling at all."""
    assert romanize("חיפה") == "khifa"


def test_romanize_vav_is_consonant_at_word_start_vowel_elsewhere():
    assert romanize("ורד") == "vrd"  # word-initial ו -> consonant 'v'
    assert romanize("מודיעין") == "modiin"  # word-internal ו -> vowel 'o'


def test_romanize_heh_is_consonant_initial_vowel_final():
    assert romanize("הרצליה") == "hrtzlia"  # initial ה='h', final ה='a'
    assert romanize("נתניה") == "ntnia"  # trailing ה -> 'a'


def test_romanize_lone_heh_word_is_definite_article_ha():
    """The special case documented on _romanize_word: after
    normalize_hebrew folds a maqaf-attached definite article ('ה־נגב')
    into a plain hyphen, the lone 'ה' token must not fall through the
    ordinary first-letter/last-letter rule (which is ambiguous for a
    single character) -- it is rendered 'ha' explicitly."""
    assert romanize("ה־נגב") == "ha-ngv"


def test_romanize_aleph_and_ayin_are_dropped():
    """Both 'usually silent' per this module's documented, accepted
    limitation -- neither has a Latin analog and dropping both is the
    deliberate, disclosed choice (see romanize()'s module comment)."""
    assert "א" not in romanize("ישראל")
    assert romanize("עכו") == "kho"  # ע (silent) + כ->kh + ו(non-initial)->o


def test_romanize_khaf_and_khet_both_map_to_kh():
    assert romanize("כפר") == "khfr"  # כ(word-initial)->kh; פ mid-word->f (see the b/p dagesh heuristic)
    assert romanize("חולון") == "kholon"  # ח -> kh


def test_romanize_tzadi_maps_to_tz():
    assert romanize("נצרת") == "ntzrt"


def test_romanize_hyphenated_compound_keeps_hyphen_and_treats_each_part_positionally():
    """'מודיעין-מכבים-רעות' is a real, officially hyphenated locality
    name. Each hyphen-separated part must be romanized as its OWN word
    for positional purposes (its own first/last letter), not as one
    long run -- otherwise only the very first letter of the whole
    compound would ever get the word-initial treatment."""
    result = romanize("מודיעין-מכבים-רעות")
    assert result == "modiin-mkhvim-rot"
    assert result.count("-") == 2


def test_romanize_final_letters_romanize_identically_to_base_forms():
    """Folding sofit->base before the letter-mapping step means a final
    letter and its base form must romanize to the same Latin letter --
    they are pronounced identically, so there is no linguistic reason
    for them to diverge, and diverging would silently reduce recall for
    real names that legitimately end in one of these five letters."""
    assert romanize("שלום") == romanize("שלומ")
    assert romanize("גן") == romanize("גנ")


def test_romanize_is_pure_and_repeatable():
    assert romanize("ירושלים") == romanize("ירושלים")


def test_romanize_empty_string_is_empty():
    assert romanize("") == ""


def test_romanize_passes_through_non_hebrew_text_unchanged_in_structure():
    """A pure-Latin string handed to romanize() should come back
    recognizably itself (this module optimizes for recall, not for
    being a no-op, but it must not crash or mangle non-Hebrew input --
    entity_resolution.py relies on romanize() being safe to call on
    mixed-script text)."""
    assert romanize("Tel Aviv") == "Tel Aviv"


def test_romanize_recall_into_fuzzy_matcher_for_common_names():
    """This is the load-bearing property, tested at the level that
    actually matters: romanize() does not need to reproduce the
    'official' English spelling character-for-character, it needs to
    land CLOSE ENOUGH that entity_resolution's Jaro-Winkler scoring
    treats it as a strong match against the real English name. Several
    of these (Haifa, Rehovot, Holon, Ashdod, Ashkelon) have no curated
    entry in TRANSLITERATION_VARIANTS, so this is exercising the
    algorithmic fallback alone, not the hand-curated table."""
    from app.entity_resolution import MIN_RELEVANCE, score_pair

    strong_pairs = [
        ("Haifa", "חיפה"),
        ("Ashdod", "אשדוד"),
        ("Rehovot", "רחובות"),
        ("Holon", "חולון"),
        ("Ashkelon", "אשקלון"),
        ("Lod", "לוד"),
    ]
    for english, hebrew in strong_pairs:
        confidence, _ = score_pair(english, hebrew)
        assert confidence >= MIN_RELEVANCE, f"{english!r} vs {hebrew!r} scored {confidence:.3f}"


# ── TRANSLITERATION_VARIANTS ──────────────────────────────────────────────
def test_transliteration_variants_covers_required_localities():
    required_keys = {
        "תל אביב-יפו",
        "ירושלים",
        "חיפה",
        "ראשון לציון",
        "פתח תקווה",
        "באר שבע",
        "כפר סבא",
        "רעננה",
        "הרצליה",
        "נתניה",
        "מודיעין-מכבים-רעות",
        "עכו",
        "נצרת",
        "אשדוד",
        "אשקלון",
        "רחובות",
        "חולון",
        "בת ים",
        "רמת גן",
        "גבעתיים",
        "כפר יונה",
    }
    assert required_keys <= TRANSLITERATION_VARIANTS.keys()


def test_transliteration_variants_covers_the_kiryat_family_both_spellings():
    """Both קרית and קריית spellings occur in real official Israeli
    government data for the same localities -- see the comment in
    hebrew.py. Both must be present so whichever spelling the real
    catalog uses, its variants are available."""
    kiryat_localities = [
        "אתא",
        "ביאליק",
        "ים",
        "מוצקין",
        "גת",
        "מלאכי",
        "שמונה",
        "אונו",
    ]
    for suffix in kiryat_localities:
        assert f"קרית {suffix}" in TRANSLITERATION_VARIANTS, f"missing קרית {suffix}"
        assert f"קריית {suffix}" in TRANSLITERATION_VARIANTS, f"missing קריית {suffix}"
        # Both spellings must carry the SAME English variants -- they name
        # the same place, just spelled two ways in the source data.
        assert TRANSLITERATION_VARIANTS[f"קרית {suffix}"] == TRANSLITERATION_VARIANTS[f"קריית {suffix}"]


def test_transliteration_variants_tel_aviv_includes_hebrew_abbreviation():
    """ת"א (the gershayim-marked Hebrew abbreviation for Tel Aviv) is
    exactly the kind of thing an Israeli user types instead of the full
    name -- it must be one of the curated variants, not just the
    English spellings."""
    assert 'ת"א' in TRANSLITERATION_VARIANTS["תל אביב-יפו"]


def test_transliteration_variants_values_are_tuples_of_strings():
    """Type discipline: the public contract is dict[str, tuple[str,
    ...]], not e.g. lists (which are mutable and would let a caller
    accidentally mutate the shared module-level table)."""
    for value in TRANSLITERATION_VARIANTS.values():
        assert isinstance(value, tuple)
        assert all(isinstance(v, str) for v in value)


# ── REGION_GROUPS / match_region ──────────────────────────────────────────
def test_region_groups_covers_required_regions():
    required = {"Gush Dan", "the Krayot", "the Sharon", "Gush Etzion", "the Negev", "the Galilee"}
    assert required <= REGION_GROUPS.keys()


def test_region_groups_krayot_membership_is_the_four_kiryat_localities():
    krayot = REGION_GROUPS["the Krayot"]
    assert set(krayot) == {"קרית אתא", "קרית ביאליק", "קרית ים", "קרית מוצקין"}


def test_region_groups_values_are_multi_member_tuples():
    """A region group with only one member wouldn't be a region-vs-
    locality ambiguity at all -- every group here must genuinely name
    more than one locality, or the whole mechanism has nothing to do."""
    for region, members in REGION_GROUPS.items():
        assert len(members) > 1, f"{region} has too few members to be a region group"


def test_match_region_hits_on_english_canonical_name():
    result = match_region("Gush Dan")
    assert result is not None
    name, members = result
    assert name == "Gush Dan"
    assert members == REGION_GROUPS["Gush Dan"]


def test_match_region_hits_on_hebrew_name():
    result = match_region("גוש דן")
    assert result is not None
    assert result[0] == "Gush Dan"


def test_match_region_is_case_insensitive_for_latin_input():
    assert match_region("gush dan") is not None
    assert match_region("GUSH DAN") is not None


def test_match_region_hits_on_curated_english_nickname():
    """'Krayot'/'the Krayot' and 'Sharon'/'the Sharon' are both real
    things people type -- with and without the leading article."""
    assert match_region("Krayot") is not None
    assert match_region("the Krayot") is not None
    assert match_region("Sharon") is not None


def test_match_region_hits_on_hebrew_with_niqqud():
    """Region names must be recognized through the same niqqud-agnostic
    normalization as locality names -- a region nickname is not a
    special case exempt from that."""
    assert match_region("הַנֶּגֶב") is not None


def test_match_region_returns_none_for_a_single_locality():
    """The whole point of match_region: a query naming ONE locality
    (not a region containing several) must not match -- that's
    entity_resolution's job, not this function's."""
    assert match_region("Tel Aviv") is None
    assert match_region("תל אביב") is None


def test_match_region_returns_none_for_junk():
    assert match_region("asdfgh") is None
    assert match_region("זזזזז") is None


def test_match_region_returns_none_for_empty_string():
    assert match_region("") is None
    assert match_region("   ") is None


def test_match_region_is_pure_and_repeatable():
    assert match_region("Gush Dan") == match_region("Gush Dan")


def test_region_name_aliases_all_point_at_a_real_region_groups_key():
    """Every alias in the lookup table must resolve to a key that
    actually exists in REGION_GROUPS -- a dangling alias would silently
    KeyError inside match_region rather than failing a review."""
    for canonical in REGION_NAME_ALIASES.values():
        assert canonical in REGION_GROUPS
