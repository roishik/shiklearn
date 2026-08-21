"""
entity_resolution.py — deterministic fuzzy matching that turns a free-text
query ("LA", "the Anchorage one", "Ankorage") into item ids, so the LLM
never has to know, remember, or guess an id itself.

RULES FOR THIS FILE (same discipline as scoring.py):
  1. ZERO I/O, ZERO LLM calls, ZERO network. Pure string comparison.
  2. Every function is PURE: same inputs -> same outputs.
  3. No third-party dependencies — Jaro-Winkler and Soundex are
     implemented here, in full, deliberately. See "Why not embeddings"
     and "Why hand-rolled" below.

WHY NOT EMBEDDINGS / word2vec (asked in review 2026-08-17, and the
answer is not "we ran out of time"):
Embeddings measure MEANING similarity. Entity resolution needs IDENTITY
similarity, and for proper nouns and codes the two actively conflict.
`LAX` and `LGB` are both rare tokens that embed as "airport-ish," so an
embedding model scores them SIMILAR — which is precisely backwards, since
telling them apart is the whole job. Production record-linkage systems
(Fellegi-Sunter, Splink, Dedupe) use string metrics + phonetic keys +
blocking for exactly this reason. Where semantics genuinely helps is the
paraphrase hop ("the windy city" -> "Chicago"), and the LLM already does
that for free upstream of this module: the model proposes a NAME, this
module resolves that name to an ID with a confidence. Model does world
knowledge; deterministic code does identity. That split is also what
keeps "the LLM can never invent an id" true.

WHY HAND-ROLLED rather than rapidfuzz/jellyfish:
Two fewer dependencies, and every line here is short enough to explain
and to unit-test in isolation. A library whose internals are opaque is a
liability when the resolver returns a surprising match and you need to
say why. Double Metaphone would beat Soundex on hard phonetics; that's a
known, accepted limitation, not an oversight.

This module's job is not merely "here are some candidates" — it's "is
this match good enough to ACT on" (`decisive` on ResolutionResult).
app/tools.py wires `decisive` straight to system-prompt behavior: when
it's False the model must state the ambiguity and ask rather than
silently pick the top candidate. That is the runtime demonstration of
"communicate assumptions and uncertainty," not just a doc asserting it.

HEBREW. This module's queries and catalog now routinely contain Hebrew
script (Israeli locality names), and users type them in Hebrew, in
English, or in any of several competing transliterations. The fuzzy
metrics above (Jaro-Winkler, Soundex, token containment) are all
script-agnostic string comparisons and need no change to run on Hebrew
text — but Soundex's coding table IS English-specific (see
app/hebrew.py's module docstring for why), so a Hebrew query never gets
useful signal from the phonetic term unless it's first romanized. See
`score_pair` below for how the Hebrew-normalized and romanized forms are
folded in alongside the plain-Latin comparison, and see
app/hebrew.match_region for the region-group concept ("Gush Dan" / "the
Krayot" name several localities, not one) — the analog of the old
project's METRO_AIRPORTS/resolve_metro layer. That concept intentionally
lives OUTSIDE this module (in app/hebrew.py, consumed by app/dataset.py
and app/tools.py) rather than inside `resolve()`, because `resolve()`'s
signature — `resolve(query, catalog, top_k)` — has no way to receive a
"region -> member localities" mapping without changing that signature,
and it must not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.hebrew import is_hebrew, normalize_hebrew, romanize

# A candidate must clear this blended score to be surfaced at all —
# below it, it isn't a plausible candidate, it's noise. NOT picked by
# feel: it's the midpoint of the measured gap between labelled
# should-match and should-not-match pairs. Re-derive it with
# `PYTHONPATH=. python scripts/calibrate_resolver.py` after touching any
# weight or metric here, and re-derive it again against REAL names once a
# this is pointed at a different real dataset — a threshold calibrated on placeholder
# data is only as good as that data.
MIN_RELEVANCE = 0.63

# The two-part bar for "decisive". Both must hold — each targets a
# different failure mode; see resolve()'s docstring.
#
# The confidence floor sits deliberately ABOVE MIN_RELEVANCE. If they
# were equal (or inverted) every surfaced candidate would clear the floor
# automatically and only the gap check would do any work — a dead
# condition that still looks like a safeguard, which is worse than not
# having it. Surfacing and acting are two different bars: 0.63 says
# "worth showing the user", 0.75 says "safe to act on unasked".
DECISIVE_MIN_CONFIDENCE = 0.75
DECISIVE_MIN_GAP = 0.15  # ...and clearly ahead of the runner-up

# Blend weights for the final confidence, CALIBRATED against labelled
# should-match / should-not-match pairs rather than picked by feel — see
# scripts/calibrate_resolver.py, which prints the separation between the
# two classes and fails loudly if they overlap.
#
# The first attempt weighted whole-string Jaro-Winkler at 0.6 and had
# NEGATIVE separation: 'the weather in paris' scored 0.54 against 'the
# alpha option' while the genuine match 'alpha' scored only 0.48. Two
# compounding causes, both worth knowing:
#   1. A short query is punished by length mismatch against a longer
#      alias, even when it appears in that alias verbatim.
#   2. A shared leading stopword ("the ") collects an undeserved Winkler
#      prefix bonus AND a free Jaccard hit.
# token_containment fixes both: it matches each query token against its
# best candidate token, weighted by token LENGTH, so long distinctive
# words dominate and stopwords can no longer carry a match on their own.
# That's why it now carries the largest share.
_W_TOKEN_CONTAINMENT = 0.5
_W_JARO_WINKLER = 0.3
_W_TOKEN_SET = 0.1
_W_PHONETIC = 0.1


@dataclass(frozen=True)
class EntityCandidate:
    item_id: str
    matched_text: str  # whichever id/name/alias produced this score
    confidence: float  # 0..1 blended score
    signals: tuple[tuple[str, float], ...]  # per-metric breakdown, same "explain, don't assert" habit as scoring.py


@dataclass(frozen=True)
class ResolutionResult:
    query: str
    candidates: tuple[EntityCandidate, ...]  # descending confidence, ties by item_id
    decisive: bool


def _normalize(text: str) -> str:
    """Casefold, collapse whitespace, drop punctuation. Deliberately does
    NOT strip generic domain words ("neighborhood", "municipality") —
    that's a per-domain decision the caller makes when building its
    catalog, not something a generic matcher should assume.

    Script-agnostic: `str.isalnum()`/`str.casefold()` both work on Hebrew
    exactly the way they work on Latin (Hebrew letters are alphabetic,
    Hebrew has no case so casefold is a no-op on them), so this function
    is unchanged from before Hebrew support existed and needs no Hebrew
    special-casing of its own — Hebrew-specific canonicalization (niqqud,
    final letters, geresh/maqaf) is a SEPARATE, prior step handled by
    app.hebrew.normalize_hebrew, not something this generic function
    should know about."""
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.casefold())
    return " ".join(cleaned.split())


def _exact_key(text: str) -> str:
    """Comparison key for EXACT (not fuzzy) matches — used only below
    the fuzzy floor in `resolve()`, where the whole point is that
    nothing forgives a spelling difference except this one normalization
    step. Hebrew text is run through normalize_hebrew first so a query
    typed without niqqud still exactly-matches a catalog alias stored
    with niqqud (or vice versa), and a final-letter spelling variant
    still counts as the same string. A no-op beyond casefold for Latin
    text, so short-code exact matching is unchanged from before Hebrew
    support existed."""
    if is_hebrew(text):
        text = normalize_hebrew(text)
    return text.strip().casefold()


def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro similarity: proportion of matching characters (within a
    sliding window) adjusted for transpositions. The base metric that
    Jaro-Winkler refines."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Characters farther apart than this can't count as a match.
    match_window = max(len1, len2) // 2 - 1
    if match_window < 0:
        match_window = 0

    s1_matched = [False] * len1
    s2_matched = [False] * len2
    matches = 0
    for i, ch in enumerate(s1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)
        for j in range(start, end):
            if not s2_matched[j] and s2[j] == ch:
                s1_matched[i] = True
                s2_matched[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Count transpositions: matched chars that appear in a different order.
    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matched[i]:
            continue
        while not s2_matched[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3


def jaro_winkler_similarity(s1: str, s2: str, prefix_scale: float = 0.1, max_prefix: int = 4) -> float:
    """Jaro, boosted when strings share a leading prefix. The prefix bonus
    is why this beats plain edit distance on names and codes: people get
    the START of a name right far more often than the end, so agreement
    there is stronger evidence of identity."""
    jaro = jaro_similarity(s1, s2)
    prefix_len = 0
    for a, b in zip(s1[:max_prefix], s2[:max_prefix]):
        if a != b:
            break
        prefix_len += 1
    return jaro + prefix_len * prefix_scale * (1 - jaro)


_SOUNDEX_CODES = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    **dict.fromkeys("l", "4"),
    **dict.fromkeys("mn", "5"),
    **dict.fromkeys("r", "6"),
}


def soundex(word: str) -> str:
    """Classic Soundex: first letter + 3 consonant-class digits, so
    same-sounding spellings collide ('Anchorage'/'Ankorage' -> A526).
    Known limitation vs. Double Metaphone: Soundex is anglocentric and
    only keys off the first letter, so it misses same-sounding words that
    start differently ('Kelly'/'Celly'). Accepted, not overlooked."""
    letters = [ch for ch in word.casefold() if ch.isalpha()]
    if not letters:
        return ""

    first = letters[0]
    encoded = [_SOUNDEX_CODES.get(ch, "") for ch in letters]

    # Collapse runs of the same code, including one that starts at the
    # first letter (its digit is dropped, but it still suppresses a
    # duplicate immediately after it).
    result = ""
    previous = encoded[0]
    for code in encoded[1:]:
        if code and code != previous:
            result += code
        # Vowels (empty code) reset the run; h/w do not, but Soundex's
        # h/w rule is a refinement this deliberately-simple version skips.
        previous = code
        if len(result) == 3:
            break

    return (first.upper() + result).ljust(4, "0")


def _token_set_similarity(a: str, b: str) -> float:
    """Jaccard overlap of word sets — order-independent, so 'Los Angeles
    International' and 'International Los Angeles' score 1.0 where a
    character metric would punish the reordering."""
    tokens_a, tokens_b = set(a.split()), set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _token_containment(query: str, candidate: str) -> float:
    """For each QUERY token, its best Jaro-Winkler match among the
    candidate's tokens, averaged with each token weighted by its own
    length.

    Length-weighting is the load-bearing part. A short query token that
    matches perfectly ('alpha' inside 'the alpha option') should score
    high; a shared stopword ('the') should contribute almost nothing,
    because three characters of agreement is weak evidence of identity
    while seven characters is strong. This deliberately avoids a
    hard-coded stopword list — those are per-language and per-domain, and
    would be one more thing to get wrong on an unfamiliar dataset.

    Asymmetric on purpose: it asks "is the query accounted for by the
    candidate", not the reverse, so a short query still matches a long
    official name without being penalised for the extra words.
    """
    query_tokens = query.split()
    candidate_tokens = candidate.split()
    if not query_tokens or not candidate_tokens:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for qt in query_tokens:
        best = max(jaro_winkler_similarity(qt, ct) for ct in candidate_tokens)
        weight = float(len(qt))
        weighted_sum += best * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight else 0.0


def _blend(q: str, c: str) -> tuple[float, tuple[tuple[str, float], ...]]:
    """The actual four-signal blend, over two ALREADY-NORMALIZED strings.
    Factored out of `score_pair` so it can be run more than once per
    query/candidate pair — once on the plain-Latin-normalized forms,
    and again on Hebrew-normalized and/or romanized forms — without
    duplicating the blend math or its weights. `score_pair` picks the
    best of however many variants it runs; this function has no opinion
    about variants at all, it just scores two strings."""
    containment = _token_containment(q, c)
    jw = jaro_winkler_similarity(q, c)
    token_set = _token_set_similarity(q, c)
    phonetic = 1.0 if soundex(q.replace(" ", "")) == soundex(c.replace(" ", "")) else 0.0

    blended = (
        _W_TOKEN_CONTAINMENT * containment
        + _W_JARO_WINKLER * jw
        + _W_TOKEN_SET * token_set
        + _W_PHONETIC * phonetic
    )
    signals = (
        ("token_containment", round(containment, 4)),
        ("jaro_winkler", round(jw, 4)),
        ("token_set", round(token_set, 4)),
        ("phonetic", phonetic),
    )
    return blended, signals


_ZERO_SIGNALS: tuple[tuple[str, float], ...] = (
    ("token_containment", 0.0),
    ("jaro_winkler", 0.0),
    ("token_set", 0.0),
    ("phonetic", 0.0),
)


def score_pair(query: str, candidate_text: str) -> tuple[float, tuple[tuple[str, float], ...]]:
    """Blend the four signals into one confidence, and return the
    per-signal breakdown alongside it — same discipline as scoring.py's
    ComponentBreakdown: never hand back a bare number nobody can explain.
    A candidate that scores 0.71 should be answerable with WHICH signal
    got it there.

    HEBREW: takes the BEST of up to four comparisons, not just one:
      1. plain-Latin-normalized (the original, unchanged path)
      2. Hebrew-normalized (niqqud/final-letters/geresh folded on
         whichever side has Hebrew — catches spelling variation within
         Hebrew script, e.g. with vs. without niqqud)
      3. query romanized, compared against the candidate as-is (catches
         an ENGLISH query hitting a Hebrew-only alias)
      4. candidate romanized, compared against the query as-is (catches
         a HEBREW query hitting a Latin-only alias)
    Variant (1) alone is computed whenever NEITHER side contains Hebrew,
    and nothing else runs — this is the fast path, and it is exactly the
    pre-Hebrew code path (same function, same inputs, same output), which
    is what guarantees English-vs-English scoring is bit-for-bit
    unchanged from before Hebrew support existed. The extra variants only
    run when at least one side is Hebrew, so a pure-Latin catalog and a
    pure-Latin query never pay for romanization or Hebrew normalization
    they don't need.

    Taking the MAX across variants (rather than, say, averaging them) is
    deliberate: these are different HYPOTHESES about which script/form
    the query and candidate are best compared in, not independent
    evidence to be combined — a Hebrew query compared against its own
    romanization of a Latin candidate is not additional confirmation on
    top of comparing the raw forms, it's an alternative reading, and the
    best-supported reading should win.
    """
    q, c = _normalize(query), _normalize(candidate_text)
    if not q or not c:
        return 0.0, _ZERO_SIGNALS

    best = _blend(q, c)

    query_has_hebrew = is_hebrew(query)
    candidate_has_hebrew = is_hebrew(candidate_text)
    if not query_has_hebrew and not candidate_has_hebrew:
        return best  # fast path: identical to the pre-Hebrew implementation

    # Variant 2: Hebrew-normalized forms (niqqud/final-letter/geresh
    # folding). normalize_hebrew() is a safe no-op on a side that has no
    # Hebrew characters (see its docstring), so this is safe to run even
    # when only ONE side is Hebrew — it just won't change the other side.
    hq, hc = _normalize(normalize_hebrew(query)), _normalize(normalize_hebrew(candidate_text))
    if hq and hc:
        candidate_score = _blend(hq, hc)
        if candidate_score[0] > best[0]:
            best = candidate_score

    # Variant 3: cross-script, query side romanized. Only worth trying
    # when the query IS Hebrew and the candidate is NOT — if the
    # candidate were also Hebrew, romanizing only one side would compare
    # scripts that don't match either way and just waste a comparison.
    if query_has_hebrew and not candidate_has_hebrew:
        rq = _normalize(romanize(query))
        if rq:
            candidate_score = _blend(rq, c)
            if candidate_score[0] > best[0]:
                best = candidate_score

    # Variant 4: cross-script, candidate side romanized (the mirror of
    # variant 3 — an English query against a Hebrew-only alias).
    if candidate_has_hebrew and not query_has_hebrew:
        rc = _normalize(romanize(candidate_text))
        if rc:
            candidate_score = _blend(q, rc)
            if candidate_score[0] > best[0]:
                best = candidate_score

    # Both sides Hebrew: also try both romanized. Cheap, and catches the
    # case where a Hebrew query has a typo that only aligns with a Hebrew
    # candidate once both are flattened into the same Latin letter
    # inventory (Hebrew Jaro-Winkler treats every letter as equally
    # distinct; romanized forms can share more surface characters).
    if query_has_hebrew and candidate_has_hebrew:
        rq, rc = _normalize(romanize(query)), _normalize(romanize(candidate_text))
        if rq and rc:
            candidate_score = _blend(rq, rc)
            if candidate_score[0] > best[0]:
                best = candidate_score

    return best


# Below this length a query is too short for fuzzy matching to mean
# anything, and it must match an alias EXACTLY (case-insensitively) to
# count.
#
# Found on real data, not anticipated: 'LA' scored 0.83 against Lawton,
# Oklahoma ('LAW'), 0.79 against La Crosse and 0.73 against Lafayette,
# because a two-character query is a prefix of all three and
# Jaro-Winkler pays a large prefix bonus. Those are not plausible
# readings of 'LA' — they are artifacts of the string being short. A
# short query carries too little signal to rank on, so the honest
# behaviour is to demand an exact hit and otherwise return nothing,
# letting `decisive=False` route it to a clarifying question.
MIN_FUZZY_QUERY_LENGTH = 4

# Hebrew gets its OWN, shorter floor, because 4 characters is tuned for
# Latin and silently breaks real Hebrew place names. Several genuine
# Israeli localities are exactly 3 characters: לוד (Lod) and עכו (Akko)
# both appear in this domain's own curated variant table
# (app.hebrew.TRANSLITERATION_VARIANTS). Under the Latin floor, a user
# typing "לוד" verbatim — not a typo, the CORRECT full name — would be
# forced through the exact-match-only branch below and get zero fuzzy
# tolerance for something as small as a stray extra space.
#
# 3, not lower. This is the deliberate tradeoff the house style asks to
# be named explicitly: 3 is the shortest REAL locality name known in this
# domain, so the floor is set at the minimum needed to admit genuine
# names — not lower "to be safe". Dropping to 2 would let common
# 2-character Hebrew fragments (prepositions, prefixed short words) into
# fuzzy scoring against every catalog alias, buying no additional real
# recall (no known 2-character Israeli locality name) while reopening
# exactly the 'LA vs Lawton/La Crosse' short-query noise problem
# MIN_FUZZY_QUERY_LENGTH exists to close off, this time in Hebrew.
MIN_FUZZY_QUERY_LENGTH_HEBREW = 3


def _min_fuzzy_query_length(text: str) -> int:
    """Which floor applies to THIS query — Hebrew and Latin queries are
    held to different bars for the reasons documented on each constant.
    A single mixed-script query (rare, but not impossible — a Hebrew name
    with a Latin qualifier) is treated as Hebrew: the Hebrew floor is the
    more permissive of the two, and it is more honest to under-restrict a
    mixed query than to force a real short Hebrew name through the
    stricter Latin gate just because it happens to be paired with a
    Latin character somewhere in the same string."""
    return MIN_FUZZY_QUERY_LENGTH_HEBREW if is_hebrew(text) else MIN_FUZZY_QUERY_LENGTH


# Short, fixed-length domain codes (IATA/FAA LocID) get a narrow exception
# to the exact-match-only rule above. A single-letter typo or adjacent-
# letter swap in a 3-4 character code ('LBG' for the real 'LGB') is a
# common, plausible query, and — restricted to comparing codes against
# other codes, never against names — catching it does not reopen the "LA"
# vs Lawton/La Crosse problem MIN_FUZZY_QUERY_LENGTH exists to stop: that
# problem came from a short query's prefix matching deep into a much
# longer multi-word name, which can't happen when both sides are capped at
# 4 characters.
MAX_CODE_EDIT_DISTANCE = 1

# A code exactly one edit from the query is about as clean a signal as
# fuzzy matching gets for a fixed-length identifier convention — high
# enough to clear DECISIVE_MIN_CONFIDENCE on its own when nothing else is
# close, so a typo'd code resolves the same way a correct one does.
_CODE_EDIT_CONFIDENCE = 0.80


def _is_code_query_shaped(text: str) -> bool:
    """3-4 characters, one token, letters/digits only — the shape of an
    IATA code or FAA LocID as a user would type it, case regardless (people
    don't bother capitalizing codes)."""
    return 3 <= len(text) <= 4 and " " not in text and text.isalnum()


def _is_code_alias(text: str) -> bool:
    """Catalog aliases that are actually codes, not names that happen to be
    short. Codes come straight from the locid/iata_code columns and are
    conventionally upper-case; names and municipalities are not (real data
    check: 552/570 short single-token aliases are upper-case codes, the
    other 18 are city names like 'Reno' and 'Waco'). Requiring upper-case
    excludes those names from the code-only comparison pool without this
    generic resolver needing any airport-specific knowledge of which alias
    came from which column."""
    return 3 <= len(text) <= 4 and " " not in text and text.isalnum() and text.isupper()


def _restricted_edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein distance, optimal-string-alignment variant:
    insertion, deletion, substitution, or one transposition of two adjacent
    characters, each cost 1. Full-matrix DP — callers only ever pass
    4-character-or-shorter codes, so there's no performance reason to trim
    it further."""
    len_a, len_b = len(a), len(b)
    d = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        d[i][0] = i
    for j in range(len_b + 1):
        d[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition
    return d[len_a][len_b]


def resolve(query: str, catalog: Mapping[str, Sequence[str]], top_k: int = 5) -> ResolutionResult:
    """`catalog` maps item_id -> every text that should count as a name
    for it (its id, display name, aliases — the caller decides scope).
    Returns candidates clearing MIN_RELEVANCE, sorted descending, plus a
    `decisive` verdict.

    `decisive` requires BOTH halves of the bar:
      - the top candidate is a solid match on its own
        (confidence >= DECISIVE_MIN_CONFIDENCE), and
      - it is clearly ahead of the runner-up
        (top - runner_up >= DECISIVE_MIN_GAP).

    Either half failing means "do not act on this without stating the
    assumption or asking." A high score with a close runner-up is the
    'LA' case — LAX vs. the basin, several readings all plausible. A lone
    low score with no competition is 'nothing here really matches', which
    is not confident just because it is unopposed. Zero candidates is
    never decisive by construction: an empty top-1 has no confidence to
    check.
    """
    # Short queries: exact alias match only. See MIN_FUZZY_QUERY_LENGTH /
    # MIN_FUZZY_QUERY_LENGTH_HEBREW — Hebrew and Latin queries use
    # different floors, decided per-query by _min_fuzzy_query_length.
    stripped = query.strip()
    if len(stripped) < _min_fuzzy_query_length(stripped):
        # Hebrew-aware exact-match key: niqqud/final-letter/geresh
        # spelling variants still count as the SAME string even below
        # the fuzzy floor, where nothing else forgives spelling
        # differences at all. _exact_key is a no-op for Latin text (same
        # casefold comparison as before Hebrew support existed).
        query_key = _exact_key(stripped)
        # ONE candidate per ITEM, not one per matching alias.
        #
        # An item legitimately carries several names that can collapse to
        # the same exact key: name_en, name_he, and the curated
        # transliteration variants all live in the catalog, and for a short
        # name several of them are the same string once _exact_key has
        # folded case, niqqud and final letters. Emitting one candidate per
        # matching alias made an item compete with ITSELF.
        #
        # That was not cosmetic. `decisive` below is a statement about how
        # many distinct PLACES the query could mean, so counting aliases
        # made a perfectly unambiguous query look ambiguous: "Lod" matched
        # locality 7000 twice at confidence 1.0 each, len(candidates) == 2,
        # decisive=False — and the agent, obeying NEVER_INVENT_IDS_RULE,
        # would then stop and ask the user to disambiguate Lod from Lod.
        # A real user-facing failure, found by scripts/calibrate_resolver.py
        # as a false negative on a 3-character name.
        #
        # First alias wins as the representative text. They are equal under
        # _exact_key by construction, so the choice only affects which
        # spelling is echoed back, and catalog order puts the display name
        # first.
        seen_items: set[str] = set()
        exact: list[EntityCandidate] = []
        for item_id, texts in catalog.items():
            for text in texts:
                if _exact_key(text) != query_key or item_id in seen_items:
                    continue
                seen_items.add(item_id)
                exact.append(
                    EntityCandidate(
                        item_id=item_id,
                        matched_text=text,
                        confidence=1.0,
                        signals=(("exact_short_query_match", 1.0),),
                    )
                )
        if exact:
            exact.sort(key=lambda c: c.item_id)
            top_exact = tuple(exact[:top_k])
            return ResolutionResult(query=query, candidates=top_exact, decisive=len(top_exact) == 1)

        if not _is_code_query_shaped(stripped):
            return ResolutionResult(query=query, candidates=(), decisive=False)

        # No exact hit, but the query looks like a code. See
        # MAX_CODE_EDIT_DISTANCE for why a one-edit near match is still
        # worth surfacing.
        normalized_query = stripped.casefold()
        near: list[EntityCandidate] = []
        for item_id, texts in catalog.items():
            best: EntityCandidate | None = None
            for text in texts:
                if not _is_code_alias(text):
                    continue
                distance = _restricted_edit_distance(normalized_query, text.casefold())
                if distance > MAX_CODE_EDIT_DISTANCE:
                    continue
                # NOTE: every candidate from this branch carries the same
                # fixed confidence, so the second test compares a constant
                # with itself and is never true — this is `best is None` in
                # effect, i.e. first match per item wins. Left as-is
                # deliberately: preferring the smaller edit distance instead
                # changes which alias represents an item, and the surfacing
                # is correct either way because same-distance codes all tie
                # at 0.80 and the decisive gate then refuses to pick one.
                if best is None or _CODE_EDIT_CONFIDENCE > best.confidence:
                    best = EntityCandidate(
                        item_id=item_id,
                        matched_text=text,
                        confidence=_CODE_EDIT_CONFIDENCE,
                        signals=(("code_edit_distance", float(distance)),),
                    )
            if best is not None:
                near.append(best)

        near.sort(key=lambda c: (-c.confidence, c.item_id))
        top_near: tuple[EntityCandidate, ...] = tuple(near[:top_k])
        if len(top_near) == 0:
            decisive = False
        else:
            runner_up = top_near[1].confidence if len(top_near) > 1 else 0.0
            decisive = (
                top_near[0].confidence >= DECISIVE_MIN_CONFIDENCE
                and (top_near[0].confidence - runner_up) >= DECISIVE_MIN_GAP
            )
        return ResolutionResult(query=query, candidates=top_near, decisive=decisive)

    scored: list[EntityCandidate] = []
    for item_id, texts in catalog.items():
        best: EntityCandidate | None = None
        for text in texts:
            confidence, signals = score_pair(query, text)
            if best is None or confidence > best.confidence:
                best = EntityCandidate(
                    item_id=item_id, matched_text=text, confidence=round(confidence, 4), signals=signals
                )
        if best is not None and best.confidence >= MIN_RELEVANCE:
            scored.append(best)

    scored.sort(key=lambda c: (-c.confidence, c.item_id))
    top: tuple[EntityCandidate, ...] = tuple(scored[:top_k])

    if len(top) == 0:
        decisive = False
    else:
        runner_up = top[1].confidence if len(top) > 1 else 0.0
        decisive = top[0].confidence >= DECISIVE_MIN_CONFIDENCE and (top[0].confidence - runner_up) >= DECISIVE_MIN_GAP

    return ResolutionResult(query=query, candidates=top, decisive=decisive)
