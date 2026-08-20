"""
hebrew.py -- makes Hebrew a first-class input to entity resolution, not an
edge case bolted onto a Latin-only matcher.

RULES FOR THIS FILE (same discipline as entity_resolution.py and
scoring.py):
  1. ZERO I/O, ZERO LLM calls, ZERO network, ZERO wall-clock, ZERO
     randomness. Pure string/data transforms only.
  2. Every function is PURE: same inputs -> same outputs.
  3. No third-party dependencies (no `unidecode`, no ICU bindings) --
     same "hand-rolled and explainable" discipline as entity_resolution's
     Jaro-Winkler/Soundex. This is Israeli PLACE NAMES specifically, a
     narrow enough domain that a general-purpose transliteration library
     would buy less than the rules below, at the cost of an opaque
     dependency.

WHY THIS MODULE EXISTS SEPARATELY FROM entity_resolution.py:
entity_resolution.py's fuzzy machinery (Jaro-Winkler, Soundex) is generic
string comparison -- it doesn't know or care what alphabet it's looking
at. But Soundex specifically is an ENGLISH phonetic algorithm: its coding
table (b/f/p/v -> 1, c/g/j/k/q/s/x/z -> 2, ...) is tuned to English
consonant clusters and is meaningless applied to Hebrew letters. Rather
than invent a Hebrew phonetic algorithm from scratch (a much bigger,
much riskier undertaking -- Hebrew phonology and Israeli colloquial
pronunciation diverge from the written consonantal root system in ways
that would take real linguistic expertise to encode correctly), this
module ROMANIZES Hebrew text and hands the Latin result to the SAME
Jaro-Winkler/Soundex pipeline that already works. One well-tested fuzzy
engine, two scripts feeding it -- not two fuzzy engines to maintain.
"""
from __future__ import annotations

# ── Unicode ranges ────────────────────────────────────────────────────────
# The Hebrew Unicode block is U+0590-U+05FF. It packs three different
# kinds of thing into adjacent code points, and this module treats them
# differently on purpose:
#   - U+05D0-U+05EA: the 22 letters themselves (identity-bearing).
#   - U+0591-U+05C7: points (niqqud/vowel marks) and cantillation marks
#     (trope) -- diacritics layered ON TOP of letters, present in
#     liturgical text and almost never in how anyone types a place name.
#     Two spellings of the same name that differ only in niqqud are the
#     SAME name, so these must be stripped before comparison, not treated
#     as distinguishing characters.
#   - A handful of punctuation code points that live inside that same
#     numeric range (maqaf U+05BE, paseq U+05C0, sof pasuq U+05C3, nun
#     hafukha U+05C6) even though they are not diacritics. Maqaf is
#     common in real place data (e.g. "מודיעין-מכבים-רעות") and is
#     handled explicitly below rather than being silently deleted by a
#     blanket range-strip -- see normalize_hebrew's docstring for why
#     order of operations matters here.
_NIQQUD_RANGE = (0x0591, 0x05C7)  # points + cantillation, per spec range
_HEBREW_BLOCK_RANGE = (0x0590, 0x05FF)  # the whole Hebrew block, for is_hebrew()

# Geresh/gershayim are punctuation, not diacritics -- they mark
# abbreviations and single-letter words (ת"א = "TA", short for Tel
# Aviv), the Hebrew equivalent of an apostrophe or a "St." period. Folded
# to their ASCII look-alikes so a catalog alias typed with a "curly"
# Hebrew mark and a query typed with a straight ASCII quote still match.
_GERESH = "׳"  # ׳
_GERSHAYIM = "״"  # ״
_MAQAF = "־"  # ־ -- the Hebrew hyphen, used inside compound place names

# Final-letter (sofit) forms. Hebrew has five letters that are written
# differently only because they fall at the end of a word -- exactly the
# way English capitalizes a sentence-initial letter without it being a
# different letter. ם/מ, ן/נ, ץ/צ, ף/פ, ך/כ are pronounced identically and
# mean the same thing; the sofit form is purely a positional spelling
# convention.
_FINAL_TO_BASE = {
    "ם": "מ",
    "ן": "נ",
    "ץ": "צ",
    "ף": "פ",
    "ך": "כ",
}

# WHY FOLDING FINAL LETTERS IS SAFE (the claim a reader should demand
# justification for, per house style): folding ם->מ etc. could in
# principle be UNSAFE if some Israeli place name ended in the *base* form
# of one of these letters while a DIFFERENT place name was spelled
# identically except for a final-form letter in that same position --
# folding would then make two genuinely different names collide. That
# does not happen, for a simple orthographic reason: Hebrew spelling
# rules make the sofit form MANDATORY at the end of a word for these five
# letters and FORBIDDEN anywhere else. A word-final מ/נ/צ/פ/כ is not a
# valid Hebrew spelling at all -- it would be flagged as a typo, not
# read as a second, competing name. So folding never merges "real word A
# spelled with sofit" into "real word B spelled without it" for the
# simple reason that "real word B" spelled that way isn't a real word.
# The fold only ever collapses a name with its OWN final-form/base-form
# variant (mid-word inflections, or careless typing that drops the sofit
# convention), never two distinct names into one.


def strip_niqqud(text: str) -> str:
    """Remove Hebrew points and cantillation marks (U+0591-U+05C7).

    Deliberately does NOT special-case maqaf/paseq/sof-pasuq/nun-hafukha,
    the handful of punctuation code points that live inside this same
    numeric range -- see the module-level comment on _NIQQUD_RANGE. That
    means calling this directly on text containing a maqaf will silently
    delete it (turning "מודיעין-פ..." into "מודיעיןפ..." with no
    hyphen). normalize_hebrew() below handles maqaf FIRST, before this
    function ever runs, precisely to avoid that. Call this function
    standalone only when that punctuation is not a concern to you.
    """
    lo, hi = _NIQQUD_RANGE
    return "".join(ch for ch in text if not (lo <= ord(ch) <= hi))


def is_hebrew(text: str) -> bool:
    """True if `text` contains at least one character from the Hebrew
    Unicode block (letters, points, or Hebrew punctuation alike -- any of
    them is a strong enough signal that this string is Hebrew script).
    Used upstream (entity_resolution.py) to decide whether the
    Hebrew-normalization and romanization branches are worth running at
    all -- a pure-Latin query should never pay for them."""
    lo, hi = _HEBREW_BLOCK_RANGE
    return any(lo <= ord(ch) <= hi for ch in text)


def normalize_hebrew(text: str) -> str:
    """Canonicalize Hebrew text so that spelling variation that carries
    no identity information collapses to one form:
      - niqqud/cantillation stripped (see strip_niqqud)
      - final letters folded to base forms (ם->מ, ן->נ, ץ->צ, ף->פ, ך->כ)
      - geresh (׳) / gershayim (״) folded to ASCII ' / "
      - maqaf (־) folded to ASCII -
      - hyphens and runs of whitespace collapsed to a single form

    ORDER OF OPERATIONS MATTERS and is the reason this isn't just four
    independent str.replace calls in any sequence:
      1. Maqaf and geresh/gershayim are folded to ASCII FIRST. Maqaf
         (U+05BE) is numerically inside strip_niqqud's U+0591-U+05C7
         range (see the module-level comment), so if niqqud were stripped
         first, the maqaf would already be gone by the time this function
         got to "fold maqaf to hyphen" -- silently turning "מודיעין-פ"
         into "מודיעיןפ" with the word boundary destroyed. Folding it to
         '-' before stripping means strip_niqqud never sees it: there is
         nothing left in that position to strip.
      2. Niqqud is stripped.
      3. Final letters are folded. This must come after niqqud-stripping
         because a sofit letter can itself carry a niqqud point, and
         stripping is a range check on the RAW character, not on some
         post-fold representation.
      4. Hyphen runs and whitespace runs are each collapsed last, once
         steps 1-3 may have introduced new ASCII hyphens (from maqaf)
         that ought to collapse the same way a doubled space does.

    Hyphens are collapsed but deliberately kept DISTINCT from spaces
    (not merged into one separator class). "מודיעין-מכבים-רעות" is the
    real, official, hyphenated form of one locality's name -- collapsing
    its hyphens to spaces would erase the fact that it's conventionally
    written as a single hyphen-joined compound, which romanize() below
    relies on to treat each hyphen-separated part as its own name-part
    for positional letter rules (see romanize()'s docstring).
    """
    # Step 1: punctuation folds, before anything that could destroy them.
    text = text.replace(_MAQAF, "-").replace(_GERESH, "'").replace(_GERSHAYIM, '"')

    # Step 2: strip niqqud/cantillation (safe now -- maqaf already gone).
    text = strip_niqqud(text)

    # Step 3: fold sofit letters to their base forms.
    text = "".join(_FINAL_TO_BASE.get(ch, ch) for ch in text)

    # Step 4: collapse whitespace runs to one space and hyphen runs to
    # one hyphen, independently -- e.g. a doubled maqaf or "- -" both
    # become a single "-", and any run of spaces becomes a single " ".
    # No regex dependency needed: splitting each space-separated segment
    # on "-" and filtering empty parts collapses adjacent hyphens AND
    # trims a leading/trailing hyphen a boundary maqaf could have left
    # behind (e.g. "-מודיעין" -> "מודיעין"), then str.split() with no
    # argument collapses whitespace runs the same way. The collapsed
    # segments are filtered a SECOND time, after hyphen-collapsing, not
    # just before -- a segment that was originally just "--" (an
    # all-hyphen token, e.g. from "word  --  word") collapses to an
    # empty string, and filtering only on the pre-collapse segment would
    # let that empty string through and reintroduce the double-space bug
    # this step exists to prevent.
    collapsed_segments = ("-".join(p for p in seg.split("-") if p) for seg in text.split())
    text = " ".join(seg for seg in collapsed_segments if seg)
    return text


# ── Romanization ─────────────────────────────────────────────────────────
#
# THIS IS THE LOAD-BEARING FUNCTION OF THIS MODULE. entity_resolution.py's
# phonetic signal is Soundex, which is English-only (see the module
# docstring above). Rather than write a Hebrew phonetic algorithm, this
# romanizes Hebrew into Latin text and hands it to the SAME
# Jaro-Winkler/Soundex/token-containment pipeline that already handles
# Latin misspellings -- so "Modin" (a plausible English mistype) and a
# romanized "מודיעין" both land in one comparison space.
#
# PERFECT TRANSLITERATION IS IMPOSSIBLE, AND THIS FUNCTION DOES NOT TRY.
# Hebrew is written (mostly) without vowels; the same consonant skeleton
# can represent multiple real words depending on vowels the text doesn't
# encode, and letters like ו/י do double duty as consonants and as vowel
# markers (matres lectionis) with no reliable way to tell which, from the
# spelling alone, without a dictionary or a language model. What this
# function optimizes for is RECALL INTO THE FUZZY MATCHER, not linguistic
# correctness: the output does not need to be a textbook-correct
# transliteration, it needs to share enough characters, in enough of the
# right places, that Jaro-Winkler/token-containment scores it close to
# how an English speaker actually spelled the same name. A rendering
# that's "close enough for fuzzy matching" and wrong is a success for
# this function; a rendering that's phonetically perfect but requires
# grammatical analysis this module doesn't do is out of scope.
#
# Per-letter mapping, applied after normalize_hebrew (so niqqud is
# already gone and sofit letters are already folded to base form --
# folding is safe for romanization the same way it's safe for matching:
# ם/מ etc. sound identical, so they romanize identically).
_LETTER_MAP = {
    "א": "",  # aleph: (usually) silent -- a glottal stop with no Latin analog
    "ב": "b",  # default; overridden to 'v' mid-word, see _romanize_word
    "ג": "g",
    "ד": "d",
    "ה": "h",  # default; overridden to 'a' word-finally, see _romanize_word
    "ו": "v",  # default; overridden to 'o' when not word-initial, see below
    "ז": "z",
    "ח": "kh",
    "ט": "t",
    "י": "i",
    "כ": "kh",  # kh per spec; real pronunciation is k/kh depending on dagesh,
    # which niqqud (already stripped) is the only signal for -- kh is the
    # commoner reading and, more importantly, the one explicitly requested.
    "ל": "l",
    "מ": "m",
    "נ": "n",
    "ס": "s",
    "ע": "",  # ayin: (usually) silent, same rationale as aleph
    "פ": "p",  # default; overridden to 'f' mid-word, see _romanize_word (same
    # dagesh-at-word-start reasoning as ב -- 'Haifa' from 'חיפה' needs the
    # mid-word letter to land as 'f', not 'p')
    "צ": "tz",
    "ק": "k",
    "ר": "r",
    "ש": "sh",
    "ת": "t",
}


def _romanize_word(word: str) -> str:
    """Romanize a single already-Hebrew-normalized word (no spaces; a
    hyphen-joined compound like 'מודיעין-מכבים-רעות' is split into its
    hyphen-separated parts by the caller, each treated as its own 'word'
    for positional purposes, then rejoined with '-').

    Two letters get position-dependent readings because a single static
    mapping is wrong for them roughly half the time, and getting them
    right is worth the extra branch:

    - ו is a consonant ('v', as in the "and" prefix vav) at the start of
      a word, and a vowel ('o', the far commoner reading inside a
      geographic name -- 'Rishon', 'Modiin', 'Motzkin' all have a
      word-internal ו read as 'o') everywhere else. This is a heuristic,
      not a rule: word-internal consonantal vav exists too. It is
      chosen because geographic names lean heavily vowel-ו.
    - ב carries dagesh (hard, 'b' sound) at the start of a word far more
      often than mid-word (soft, 'v' sound) -- a real, if not
      exceptionless, pattern in Hebrew orthography that this leans on
      since niqqud (the actual dagesh marker) has already been stripped
      and is unavailable as a real signal.
    - פ gets the identical treatment for the identical reason: word-
      initial 'p' (dagesh), mid-word 'f' (no dagesh) -- 'חיפה' needs the
      mid-word rendering to land as 'f' for the result to resemble
      'Haifa' at all.
    - ה is the definite article / a genuine name-initial consonant ('h',
      as in Herzliya, HaNegev) at the start of a word, but very commonly
      a trailing vowel marker ('a', as in Netanya, Raanana, Herzliya's
      OWN ending) at the end of a word. Handled positionally rather than
      by trying to detect "is this actually the definite article", which
      would need morphological knowledge this module deliberately does
      not have.

    The special case for a word that is JUST 'ה' (one letter) covers the
    definite article ה־ when it was originally attached via maqaf to the
    next word ("ה־נגב", HaNegev) -- after normalize_hebrew folds the
    maqaf to a plain hyphen and this caller splits on hyphens, that
    definite article shows up as its own single-letter "word", which the
    ordinary first-letter/last-letter rule would render ambiguously
    (it's simultaneously first AND last). Rendered as "ha" explicitly
    instead.

    Letters NOT ן/ו/ה/ב get one static rendering everywhere -- see
    _LETTER_MAP. This module does not attempt to segment off single-
    letter prepositional prefixes (ו/ב/ל/מ/ש/כ, "and/in/to/from/that/as")
    that Hebrew attaches directly to the following word with no space or
    hyphen (e.g. "בחיפה" = "in Haifa"): doing that correctly needs a
    dictionary lookup ("is what follows a real word") that this
    zero-dependency module does not have. Left un-segmented, the
    prefix's own letter is still transliterated (by the static or
    positional rule for whatever letter it is) and the root name
    underneath still comes out as a recognizable SUBSTRING of the
    result -- 'khifa' inside romanize('בחיפה') still contains enough of
    'haifa' for token-containment scoring to find it. That is the
    recall-not-correctness bar this whole function is held to.
    """
    if word == "ה":
        return "ha"
    if not word:
        return word

    out_chars: list[str] = []
    last_index = len(word) - 1
    for i, ch in enumerate(word):
        if ch == "ו":
            out_chars.append("v" if i == 0 else "o")
        elif ch == "ב":
            out_chars.append("b" if i == 0 else "v")
        elif ch == "פ":
            out_chars.append("p" if i == 0 else "f")
        elif ch == "ה":
            out_chars.append("a" if i == last_index else "h")
        else:
            out_chars.append(_LETTER_MAP.get(ch, ch))  # non-Hebrew chars pass through unchanged
    return "".join(out_chars)


def romanize(text: str) -> str:
    """Deterministic Hebrew -> Latin transliteration, optimized for
    recall into the fuzzy matcher (see the module comment above this
    section) rather than for linguistic correctness. Non-Hebrew
    characters (spaces, digits, already-Latin text, punctuation) pass
    through unchanged, so this is also safe to call on mixed-script or
    pure-Latin input -- it degrades to a no-op rather than mangling
    text it has no business touching.
    """
    normalized = normalize_hebrew(text)
    words = normalized.split(" ")
    romanized_words = []
    for word in words:
        if not word:
            romanized_words.append(word)
            continue
        # Compound names are hyphen-joined ("מודיעין-מכבים-רעות"); each
        # hyphen-separated part is its own name-part for positional
        # purposes (its own "first letter", its own "last letter"), the
        # same way a space-separated word is.
        parts = word.split("-")
        romanized_words.append("-".join(_romanize_word(p) for p in parts))
    return " ".join(romanized_words)


# ── Hand-curated transliteration variants ────────────────────────────────
# The transliterations Israelis and English speakers actually type for
# major localities, gathered from how these places are commonly spelled
# in English-language Israeli media, Waze/Google Maps, and government
# English-language publications -- NOT derived from romanize() above.
# romanize() is a recall net for the long tail; this table is a precision
# fast-path for the places people search for constantly, where "good
# enough for fuzzy matching" isn't good enough and an exact, hand-checked
# spelling earns its place in the catalog directly.
#
# Consumed by the dataset-building layer (app/dataset.py, owned by
# another agent), which folds these into each locality's alias list --
# this module only supplies the curated mapping, it does not touch the
# catalog itself (this module has zero I/O and does not know what a
# "catalog" even is; see entity_resolution.py for that contract).
TRANSLITERATION_VARIANTS: dict[str, tuple[str, ...]] = {
    # NOTE on the "תל אביב -יפו" duplicate key just below: the real
    # dataset (data/processed_data/localities.json, built by the
    # data-pipeline agent from nadlan.gov.il) spells this locality's
    # name_he with a SPACE before the hyphen ("תל אביב -יפו"), not the
    # no-space "תל אביב-יפו" this table originally used as its only key.
    # A dict lookup by the dataset-building layer
    # (TRANSLITERATION_VARIANTS.get(row['name_he'])) is exact-string, so
    # the mismatch would silently drop every curated alias -- including
    # "Tel Aviv" itself -- for the single most important locality in the
    # catalog. Found by scripts/calibrate_resolver.py building a catalog
    # against the real file. Fixed the same way the קרית/קריית family
    # below already handles a genuine two-spelling split in official
    # data: both spellings as separate keys pointing at the identical
    # tuple, rather than betting on one canonical spelling winning.
    "תל אביב-יפו": ("Tel Aviv", "Tel Aviv-Yafo", "Tel Aviv-Jaffa", "TLV", 'ת"א'),
    "תל אביב -יפו": ("Tel Aviv", "Tel Aviv-Yafo", "Tel Aviv-Jaffa", "TLV", 'ת"א'),
    "ירושלים": ("Jerusalem", "Yerushalayim"),
    "חיפה": ("Haifa", "Hefa"),
    "ראשון לציון": ("Rishon LeZion", "Rishon Lezion", "Rishon LeTsiyon"),
    "פתח תקווה": ("Petah Tikva", "Petach Tikwa", "Petah Tiqwa"),
    "באר שבע": ("Beer Sheva", "Be'er Sheva", "Beersheba"),
    "כפר סבא": ("Kfar Saba", "Kefar Sava"),
    "רעננה": ("Raanana", "Ra'anana"),
    "הרצליה": ("Herzliya", "Herzeliya"),
    "נתניה": ("Netanya", "Nathanya"),
    # Deliberately NOT curating a bare "Modiin" here. מודיעין-מכבים-רעות
    # and מודיעין עילית (below) are two different, real localities that
    # both genuinely start with "Modiin" -- giving the bare form as an
    # exact curated alias to only ONE of them would hand it a spurious
    # 1.0 exact-match score with no competition, forcing
    # resolve("Modiin") to come back decisive=True when it must not
    # (see tests/test_entity_resolution.py's AMBIGUOUS_CATALOG comment,
    # which flags this exact hazard, and
    # test_ambiguous_prefix_modiin_is_not_decisive). Only the fuller,
    # unambiguous forms are curated; "Modiin" alone is left to fuzzy
    # matching, which correctly scores both candidates close together.
    "מודיעין-מכבים-רעות": ("Modi'in", "Modiin-Maccabim-Reut", "Modi'in-Maccabim-Re'ut"),
    "מודיעין עילית": ("Modiin Illit", "Modi'in Illit", "Modiin Ilit"),
    "עכו": ("Akko", "Acre"),
    "נצרת": ("Nazareth", "Natzrat"),
    "אשדוד": ("Ashdod",),
    "אשקלון": ("Ashkelon", "Ashqelon"),
    "רחובות": ("Rehovot", "Rechovot"),
    "חולון": ("Holon",),
    "בת ים": ("Bat Yam",),
    # The רמת ("Ramat" = "height of") family. Same hazard as Modiin above,
    # but the failure mode is subtler: it isn't a bare "Ramat" alias
    # (nobody curated one), it's an IMBALANCE -- רמת גן had a curated
    # English alias and its רמת-prefixed siblings did not, so
    # resolve("Ramat") scored רמת גן's curated "Ramat Gan" cleanly above
    # DECISIVE_MIN_CONFIDENCE with a clear gap to whatever the
    # uncurated candidates could manage on romanize() alone (romanize
    # can't recover the "a" vowels in "Ramat" from the consonant-only
    # spelling רמת, so its own fuzzy score line was always going to be
    # weaker -- see romanize()'s docstring on why this is inherent to
    # Hebrew orthography, not a bug in romanize() itself). Giving all
    # three siblings equally strong curated aliases restores a level
    # playing field, so "Ramat" alone lands close-scored across all
    # three and correctly comes back decisive=False. Found the same way
    # as the Modiin issue: building a catalog from the real dataset
    # (scripts/calibrate_resolver.py) rather than only the hand-rolled
    # AMBIGUOUS_CATALOG in tests/test_entity_resolution.py, which had
    # already anticipated this fix and encoded it as its own separate
    # fixture -- this brings the real table in line with that intent.
    "רמת גן": ("Ramat Gan",),
    "רמת השרון": ("Ramat HaSharon", "Ramat Hasharon"),
    "רמת ישי": ("Ramat Yishai", "Ramat Ishi"),
    "גבעתיים": ("Givatayim",),
    "כפר יונה": ("Kfar Yona", "Kefar Yona"),
    # The קרית/קריית family. Both spellings ("קרית", missing the second
    # yud, and "קריית", with it) occur in official Israeli government
    # data for the SAME locality -- this is a genuine orthographic split
    # in the source data, not a typo, so both spellings are listed as
    # separate keys pointing at identical variant tuples rather than
    # relying on any one canonical spelling being "the" key. Whichever
    # spelling the real catalog (app/dataset.py) actually uses as its
    # official name, the other spelling's entry here is still available
    # for the dataset builder to fold in as an extra alias.
    "קרית אתא": ("Kiryat Ata", "Qiryat Ata"),
    "קריית אתא": ("Kiryat Ata", "Qiryat Ata"),
    "קרית ביאליק": ("Kiryat Bialik", "Qiryat Bialik"),
    "קריית ביאליק": ("Kiryat Bialik", "Qiryat Bialik"),
    "קרית ים": ("Kiryat Yam", "Qiryat Yam"),
    "קריית ים": ("Kiryat Yam", "Qiryat Yam"),
    "קרית מוצקין": ("Kiryat Motzkin", "Qiryat Motzkin"),
    "קריית מוצקין": ("Kiryat Motzkin", "Qiryat Motzkin"),
    "קרית גת": ("Kiryat Gat", "Qiryat Gat"),
    "קריית גת": ("Kiryat Gat", "Qiryat Gat"),
    "קרית מלאכי": ("Kiryat Malakhi", "Qiryat Malakhi"),
    "קריית מלאכי": ("Kiryat Malakhi", "Qiryat Malakhi"),
    "קרית שמונה": ("Kiryat Shmona", "Qiryat Shmona"),
    "קריית שמונה": ("Kiryat Shmona", "Qiryat Shmona"),
    "קרית אונו": ("Kiryat Ono", "Qiryat Ono"),
    "קריית אונו": ("Kiryat Ono", "Qiryat Ono"),
}


# ── Region groups ─────────────────────────────────────────────────────────
# The analog of the old US-airport project's METRO_AIRPORTS layer (see
# app/dataset.py's `METRO_AIRPORTS`/`resolve_metro` in the seed codebase
# this project came from): a name a user might type that does NOT
# identify one locality, but a cluster of several. "Compare the Krayot"
# is a real, reasonable question the way "compare LA" was for airports,
# and none of the localities inside a region group are named "the
# Krayot" -- resolving it to any one of them would silently decide
# something the user didn't ask.
#
# Values are Hebrew locality names (matching whatever spelling the real
# catalog uses is the consuming layer's job, not this pure module's --
# see the note below REGION_NAME_ALIASES). Membership here is
# deliberately NOT exhaustive for the larger metro regions (Gush Dan, the
# Sharon, the Negev, the Galilee genuinely contain dozens of localities);
# it's illustrative enough for the consuming layer to build the true
# membership set against the real catalog, while still being directly
# usable as-is for the well-defined, small, canonical clusters (the
# Krayot, Gush Etzion) where the full membership IS this short.
REGION_GROUPS: dict[str, tuple[str, ...]] = {
    "Gush Dan": (
        "תל אביב-יפו",
        "רמת גן",
        "גבעתיים",
        "בני ברק",
        "חולון",
        "בת ים",
        "קריית אונו",
        "אור יהודה",
        "אזור",
    ),
    "the Krayot": ("קרית אתא", "קרית ביאליק", "קרית ים", "קרית מוצקין"),
    "the Sharon": ("רעננה", "הרצליה", "כפר סבא", "נתניה", "הוד השרון", "רמת השרון"),
    "Gush Etzion": ("אפרת", "אלון שבות", "כפר עציון", "נווה דניאל"),
    "the Negev": ("באר שבע", "אילת", "דימונה", "ערד", "מצפה רמון"),
    "the Galilee": ("נצרת", "עכו", "כרמיאל", "צפת", "קרית שמונה"),
}

# Every alias a user might type for a region group, mapped to that
# group's canonical key in REGION_GROUPS -- both English and Hebrew
# forms, since (per this module's whole reason for existing) a user
# might type either. Keys here are stored in NORMALIZED form (Hebrew
# side run through normalize_hebrew + casefold, Latin side casefolded)
# so match_region() below can normalize an incoming query once and do a
# single dict lookup rather than re-comparing against every spelling
# variant in a loop.
_REGION_ALIASES_RAW: dict[str, str] = {
    "gush dan": "Gush Dan",
    "גוש דן": "Gush Dan",
    "the krayot": "the Krayot",
    "krayot": "the Krayot",
    "the qrayot": "the Krayot",
    "הקריות": "the Krayot",
    "קריות": "the Krayot",
    "the sharon": "the Sharon",
    "sharon": "the Sharon",
    "השרון": "the Sharon",
    "שרון": "the Sharon",
    "gush etzion": "Gush Etzion",
    "gush etsion": "Gush Etzion",
    "גוש עציון": "Gush Etzion",
    "the negev": "the Negev",
    "negev": "the Negev",
    "הנגב": "the Negev",
    "נגב": "the Negev",
    "the galilee": "the Galilee",
    "galilee": "the Galilee",
    "הגליל": "the Galilee",
    "גליל": "the Galilee",
}


def _normalize_region_key(text: str) -> str:
    """Same normalize-then-casefold treatment for both scripts, used to
    build and to query the alias index -- Hebrew text is niqqud-stripped
    and final-letter-folded via normalize_hebrew, Latin text is just
    casefolded, and either result is casefolded again (a no-op for
    Hebrew, which has no case) so the two code paths converge on one
    comparable form."""
    if is_hebrew(text):
        text = normalize_hebrew(text)
    return text.strip().casefold()


REGION_NAME_ALIASES: dict[str, str] = {_normalize_region_key(k): v for k, v in _REGION_ALIASES_RAW.items()}


def match_region(query: str) -> tuple[str, tuple[str, ...]] | None:
    """If `query` names a REGION_GROUPS entry (in Hebrew, English, or any
    curated alias spelling), return (canonical_name, hebrew_locality_names).
    Otherwise None.

    Exact-match only, deliberately, mirroring the old project's
    resolve_metro: fuzzy-matching region names would reopen the same
    short-query noise problem entity_resolution.MIN_FUZZY_QUERY_LENGTH
    exists to close off (short region nicknames like 'the Sharon' are
    exactly the shape of query that fuzzy scoring handles worst).

    This function deliberately returns Hebrew locality NAMES, not
    catalog ids -- this module has no catalog to resolve them against
    (see the module docstring: zero I/O, and it doesn't know what a
    catalog looks like). The consuming layer (app/dataset.py /
    app/tools.py, owned by other agents, analogous to how
    dataset.resolve_metro / tools.resolve_entity's 'metro_area' branch
    consume METRO_AIRPORTS) is expected to map each name through its own
    catalog and label the result with a distinct match_type ("region" or
    similar) so the caller can state that reading rather than silently
    collapsing a region to one locality inside it.
    """
    key = _normalize_region_key(query.strip())
    if not key:
        return None
    canonical = REGION_NAME_ALIASES.get(key)
    if canonical is None:
        # Also accept the REGION_GROUPS key itself typed directly
        # (e.g. exactly "Gush Etzion"), without requiring every key to
        # be duplicated into _REGION_ALIASES_RAW.
        for group_name in REGION_GROUPS:
            if _normalize_region_key(group_name) == key:
                canonical = group_name
                break
    if canonical is None:
        return None
    return canonical, REGION_GROUPS[canonical]
