"""Language-aware processing capabilities for text research.

Central place for language codes, stopword sets, stem/lemma availability,
and sentence-boundary hints. Generic APIs must not assume English.

Unknown / unsupported languages degrade gracefully:
- Unicode tokenization still works
- stopword lists are empty (English stopwords are NOT applied)
- stemming / lemmatization report unavailable (callers must not claim them)
- sentence segmentation uses a generic Unicode boundary heuristic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import regex
import simplemma

# Re-export-friendly: BCP-47 primary subtag → Snowball algorithm name.
SNOWBALL_ALGORITHMS: dict[str, str] = {
    "en": "english",
    "eng": "english",
    "de": "german",
    "deu": "german",
    "ger": "german",
    "fr": "french",
    "fra": "french",
    "fre": "french",
    "es": "spanish",
    "spa": "spanish",
    "it": "italian",
    "ita": "italian",
    "pt": "portuguese",
    "por": "portuguese",
    "nl": "dutch",
    "nld": "dutch",
    "sv": "swedish",
    "swe": "swedish",
    "no": "norwegian",
    "nob": "norwegian",
    "da": "danish",
    "dan": "danish",
    "fi": "finnish",
    "fin": "finnish",
    "ru": "russian",
    "rus": "russian",
    "hu": "hungarian",
    "hun": "hungarian",
    "ro": "romanian",
    "ron": "romanian",
    "rum": "romanian",
}

# Unicode-aware word tokens (letters/marks/digits + internal apostrophe).
UNICODE_TOKEN_RE = regex.compile(
    r"[\p{L}\p{M}\p{N}]+(?:['’][\p{L}\p{M}]+)?",
    regex.VERSION1,
)

# Spanish/Portuguese-style inverted marks should not create spurious splits.
_IBERIAN_OPENERS = frozenset("¿¡")

_LEMMA_PROBE = "running"

# Sentence terminals incl. Latin + CJK + ellipsis; trailing closers stay with sentence.
SENTENCE_BOUNDARY_RE = regex.compile(
    r"[.!?…。！？]+[\"'”’)\]]*(?=\s|$)",
    regex.VERSION1,
)
# Back-compat alias.
GENERIC_SENTENCE_BOUNDARY_RE = SENTENCE_BOUNDARY_RE

# Abbreviations WITHOUT trailing period (matched case-insensitively).
_ABBREVIATIONS_EN = frozenset(
    {
        "e.g",
        "i.e",
        "etc",
        "vs",
        "viz",
        "cf",
        "approx",
        "dept",
        "dr",
        "mr",
        "mrs",
        "ms",
        "mssrs",
        "prof",
        "fig",
        "figs",
        "eq",
        "eqs",
        "vol",
        "vols",
        "no",
        "nos",
        "pp",
        "ch",
        "sec",
        "al",
        "ed",
        "eds",
        "rev",
        "gen",
        "lt",
        "col",
        "sgt",
        "jr",
        "sr",
        "st",
        "ave",
        "blvd",
        "rd",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "u.s",
        "u.k",
        "e.u",
        "ph.d",
        "m.d",
        "b.a",
        "m.a",
        "inc",
        "ltd",
        "co",
        "corp",
        "assn",
    }
)
_ABBREVIATIONS_DE = frozenset(
    {
        "z.b",
        "u.a",
        "u.ä",
        "d.h",
        "bzw",
        "ca",
        "dr",
        "prof",
        "usw",
        "vgl",
        "evtl",
        "inkl",
        "exkl",
        "nr",
        "abs",
        "art",
        "s",
        "bd",
        "hrsg",
    }
)
_ABBREVIATIONS_FR = frozenset(
    {
        "m",
        "mme",
        "mlle",
        "dr",
        "pr",
        "prof",
        "c.-à-d",
        "c.a.d",
        "etc",
        "cf",
        "ex",
        "fig",
        "vol",
        "p",
        "pp",
        "n",
        "ste",
        "st",
    }
)
_ABBREVIATIONS_ES = frozenset(
    {
        "dr",
        "dra",
        "sr",
        "sra",
        "srta",
        "prof",
        "etc",
        "pág",
        "pags",
        "vol",
        "fig",
        "ee.uu",
        "u.s",
        "núm",
        "cap",
    }
)
_ABBREVIATIONS_TR = frozenset(
    {
        "dr",
        "prof",
        "doç",
        "yrd",
        "vb",
        "vd",
        "örn",
        "vs",
        "no",
        "mad",
        "sf",
        "bkz",
    }
)

_ABBREVIATIONS_BY_LANG: dict[str, frozenset[str]] = {
    "en": _ABBREVIATIONS_EN,
    "eng": _ABBREVIATIONS_EN,
    "de": _ABBREVIATIONS_DE,
    "deu": _ABBREVIATIONS_DE,
    "ger": _ABBREVIATIONS_DE,
    "fr": _ABBREVIATIONS_FR,
    "fra": _ABBREVIATIONS_FR,
    "fre": _ABBREVIATIONS_FR,
    "es": _ABBREVIATIONS_ES,
    "spa": _ABBREVIATIONS_ES,
    "tr": _ABBREVIATIONS_TR,
    "tur": _ABBREVIATIONS_TR,
}

# Shared cross-language short forms always protected.
_ABBREVIATIONS_COMMON = frozenset(
    {"e.g", "i.e", "etc", "dr", "prof", "fig", "vs", "cf", "u.s", "u.k", "e.u"}
)

_TOKEN_BEFORE_PERIOD_RE = regex.compile(r"[\p{L}\p{N}.]+$", regex.VERSION1)
_ACRONYM_DOTTED_RE = regex.compile(r"^(?:[\p{L}]\.)+[\p{L}]?$", regex.VERSION1)
_SINGLE_INITIAL_RE = regex.compile(r"^[\p{L}]$", regex.VERSION1)
_NUMBERED_LIST_PREFIX_RE = regex.compile(r"^\s*\d+$")



# --- Stopword / negation inventories (small, auditable; not exhaustive) -----

_EN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "as",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "we",
        "our",
        "you",
        "your",
        "i",
        "my",
        "me",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "shall",
        "may",
        "might",
        "must",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
    }
)
_EN_NEGATION = frozenset({"not", "no", "never"})

_DE_STOPWORDS = frozenset(
    {
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einer",
        "eines",
        "und",
        "oder",
        "aber",
        "wenn",
        "dann",
        "von",
        "zu",
        "in",
        "an",
        "auf",
        "für",
        "mit",
        "als",
        "bei",
        "aus",
        "ist",
        "sind",
        "war",
        "waren",
        "sein",
        "diese",
        "dieser",
        "dieses",
        "jenes",
        "es",
        "er",
        "sie",
        "wir",
        "ihr",
        "mich",
        "mir",
        "sich",
        "auch",
        "noch",
        "nur",
        "schon",
        "sehr",
        "hier",
        "dort",
        "wie",
        "was",
        "wer",
        "wo",
        "wann",
        "warum",
    }
)
_DE_NEGATION = frozenset({"nicht", "kein", "keine", "nie", "niemals"})

_FR_STOPWORDS = frozenset(
    {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "et",
        "ou",
        "mais",
        "si",
        "alors",
        "à",
        "au",
        "aux",
        "en",
        "dans",
        "sur",
        "pour",
        "avec",
        "par",
        "comme",
        "est",
        "sont",
        "était",
        "être",
        "ce",
        "cette",
        "ces",
        "il",
        "elle",
        "ils",
        "elles",
        "nous",
        "vous",
        "je",
        "tu",
        "mon",
        "ton",
        "son",
        "leur",
        "qui",
        "que",
        "quoi",
        "où",
        "quand",
        "comment",
        "plus",
        "moins",
        "très",
        "aussi",
    }
)
_FR_NEGATION = frozenset({"ne", "pas", "non", "jamais", "rien"})

_ES_STOPWORDS = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "y",
        "o",
        "pero",
        "si",
        "de",
        "del",
        "a",
        "al",
        "en",
        "con",
        "por",
        "para",
        "como",
        "es",
        "son",
        "era",
        "ser",
        "este",
        "esta",
        "estos",
        "estas",
        "él",
        "ella",
        "ellos",
        "ellas",
        "nosotros",
        "vosotros",
        "yo",
        "tú",
        "mi",
        "tu",
        "su",
        "que",
        "quien",
        "donde",
        "cuando",
        "como",
        "muy",
        "más",
        "menos",
        "también",
    }
)
_ES_NEGATION = frozenset({"no", "nunca", "jamás", "nadie", "nada"})

_TR_STOPWORDS = frozenset(
    {
        "ve",
        "veya",
        "ile",
        "için",
        "bir",
        "bu",
        "şu",
        "o",
        "da",
        "de",
        "ki",
        "mi",
        "mı",
        "mu",
        "mü",
        "gibi",
        "kadar",
        "daha",
        "çok",
        "az",
        "en",
        "var",
        "yok",
        "ise",
        "ama",
        "fakat",
        "çünkü",
        "eğer",
        "ben",
        "sen",
        "biz",
        "siz",
        "onlar",
        "benim",
        "senin",
        "onun",
        "ne",
        "kim",
        "nerede",
        "nasıl",
        "neden",
        "hangi",
    }
)
_TR_NEGATION = frozenset({"değil", "yok", "asla", "hiç", "hayır"})

_PT_STOPWORDS = frozenset(
    {
        "o",
        "a",
        "os",
        "as",
        "um",
        "uma",
        "e",
        "ou",
        "mas",
        "se",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "com",
        "por",
        "para",
        "como",
        "é",
        "são",
        "era",
        "ser",
        "este",
        "esta",
        "ele",
        "ela",
        "nós",
        "você",
        "eu",
        "meu",
        "seu",
        "que",
        "quem",
        "onde",
        "quando",
        "muito",
        "mais",
        "também",
    }
)
_PT_NEGATION = frozenset({"não", "nunca", "jamais", "nada", "ninguém"})

_IT_STOPWORDS = frozenset(
    {
        "il",
        "lo",
        "la",
        "i",
        "gli",
        "le",
        "un",
        "una",
        "e",
        "o",
        "ma",
        "se",
        "di",
        "del",
        "della",
        "a",
        "da",
        "in",
        "con",
        "per",
        "come",
        "è",
        "sono",
        "era",
        "essere",
        "questo",
        "questa",
        "egli",
        "ella",
        "noi",
        "voi",
        "io",
        "tu",
        "mio",
        "tuo",
        "suo",
        "che",
        "chi",
        "dove",
        "quando",
        "molto",
        "più",
        "anche",
    }
)
_IT_NEGATION = frozenset({"non", "mai", "nessuno", "niente", "nulla"})

_LANGUAGE_LEXICONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "en": (_EN_STOPWORDS, _EN_NEGATION),
    "eng": (_EN_STOPWORDS, _EN_NEGATION),
    "de": (_DE_STOPWORDS, _DE_NEGATION),
    "deu": (_DE_STOPWORDS, _DE_NEGATION),
    "ger": (_DE_STOPWORDS, _DE_NEGATION),
    "fr": (_FR_STOPWORDS, _FR_NEGATION),
    "fra": (_FR_STOPWORDS, _FR_NEGATION),
    "fre": (_FR_STOPWORDS, _FR_NEGATION),
    "es": (_ES_STOPWORDS, _ES_NEGATION),
    "spa": (_ES_STOPWORDS, _ES_NEGATION),
    "tr": (_TR_STOPWORDS, _TR_NEGATION),
    "tur": (_TR_STOPWORDS, _TR_NEGATION),
    "pt": (_PT_STOPWORDS, _PT_NEGATION),
    "por": (_PT_STOPWORDS, _PT_NEGATION),
    "it": (_IT_STOPWORDS, _IT_NEGATION),
    "ita": (_IT_STOPWORDS, _IT_NEGATION),
}

_DISPLAY_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "tr": "Turkish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "ru": "Russian",
    "hu": "Hungarian",
    "ro": "Romanian",
}


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """Resolved language capabilities for a processing request."""

    code: str
    display_name: str
    known: bool
    degraded: bool
    tokenizer: str = "unicode_regex"
    stopwords: frozenset[str] = field(default_factory=frozenset)
    negation_words: frozenset[str] = field(default_factory=frozenset)
    stemming_available: bool = False
    lemmatization_available: bool = False
    snowball_algorithm: str | None = None
    sentence_segmentation: str = "generic_unicode"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "display_name": self.display_name,
            "known": self.known,
            "degraded": self.degraded,
            "tokenizer": self.tokenizer,
            "stopword_count": len(self.stopwords),
            "negation_word_count": len(self.negation_words),
            "stemming_available": self.stemming_available,
            "lemmatization_available": self.lemmatization_available,
            "snowball_algorithm": self.snowball_algorithm,
            "sentence_segmentation": self.sentence_segmentation,
            "notes": self.notes,
        }


def normalize_language_code(language: str | None) -> str | None:
    """Return primary language subtag, or ``None`` if missing/blank."""
    if language is None:
        return None
    stripped = language.strip()
    if not stripped:
        return None
    return stripped.lower().replace("_", "-").split("-", 1)[0]


def snowball_algorithm_for_language(language: str | None) -> str | None:
    code = normalize_language_code(language)
    if code is None:
        return None
    return SNOWBALL_ALGORITHMS.get(code)


def stemming_available(language: str | None) -> bool:
    return snowball_algorithm_for_language(language) is not None


def lemmatization_available(language: str | None) -> bool:
    code = normalize_language_code(language)
    if code is None:
        return False
    try:
        simplemma.lemmatize(_LEMMA_PROBE, lang=code)
        return True
    except (ValueError, KeyError, OSError):
        return False


def stopwords_for(language: str | None) -> frozenset[str]:
    """Language-specific stopwords. Unknown → empty (never English by default)."""
    code = normalize_language_code(language)
    if code is None:
        return frozenset()
    pair = _LANGUAGE_LEXICONS.get(code)
    return pair[0] if pair else frozenset()


def negation_words_for(language: str | None) -> frozenset[str]:
    code = normalize_language_code(language)
    if code is None:
        return frozenset()
    pair = _LANGUAGE_LEXICONS.get(code)
    return pair[1] if pair else frozenset()


def has_lexicon(language: str | None) -> bool:
    code = normalize_language_code(language)
    return code is not None and code in _LANGUAGE_LEXICONS


def resolve_language(language: str | None) -> LanguageProfile:
    """Resolve capabilities for ``language``; unknown codes degrade gracefully."""
    code = normalize_language_code(language)
    if code is None:
        return LanguageProfile(
            code="und",
            display_name="Undetermined",
            known=False,
            degraded=True,
            sentence_segmentation="generic_heuristic_v2",
            notes="No language configured; using Unicode-generic processing only.",
        )

    stem_algo = snowball_algorithm_for_language(code)
    lemma_ok = lemmatization_available(code)
    lexicon_ok = has_lexicon(code)
    known = lexicon_ok or stem_algo is not None or lemma_ok
    degraded = not known

    if code in {"es", "spa", "pt", "por"}:
        sentence_mode = "iberian_heuristic_v2"
    elif known:
        sentence_mode = "latin_heuristic_v2"
    else:
        sentence_mode = "generic_heuristic_v2"

    notes = ""
    if degraded:
        notes = (
            "Unknown or unsupported language; Unicode tokenization only. "
            "Stopwords/stem/lemma not applied unless explicitly available."
        )
    elif not lexicon_ok:
        notes = "Stem/lemma may be available, but no built-in stopword lexicon."

    return LanguageProfile(
        code=code,
        display_name=_DISPLAY_NAMES.get(code, code),
        known=known,
        degraded=degraded,
        stopwords=stopwords_for(code),
        negation_words=negation_words_for(code),
        stemming_available=stem_algo is not None,
        lemmatization_available=lemma_ok,
        snowball_algorithm=stem_algo,
        sentence_segmentation=sentence_mode,
        notes=notes,
    )


def tokenize_unicode(text: str) -> list[str]:
    """Language-agnostic Unicode word tokenization."""
    return UNICODE_TOKEN_RE.findall(text)


def abbreviations_for(language: str | None) -> frozenset[str]:
    """Abbreviation stems (no trailing period) for false-boundary detection."""
    code = normalize_language_code(language)
    base = set(_ABBREVIATIONS_COMMON)
    if code and code in _ABBREVIATIONS_BY_LANG:
        base.update(_ABBREVIATIONS_BY_LANG[code])
    elif code is None or code == "und":
        # Undetermined: use English + common to avoid obvious false splits.
        base.update(_ABBREVIATIONS_EN)
    return frozenset(base)


def sentence_boundary_pattern(language: str | None = None) -> regex.Pattern[str]:
    """Return sentence-boundary regex (language selects heuristics, not the pattern)."""
    _ = language
    return SENTENCE_BOUNDARY_RE


def _token_start_before(text: str, index: int) -> int:
    start = index
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return start


def _token_end_after(text: str, index: int) -> int:
    end = index
    while end < len(text) and not text[end].isspace():
        end += 1
    return end


def is_false_sentence_boundary(
    text: str,
    match_start: int,
    match_end: int,
    language: str | None = None,
) -> bool:
    """Return True when a candidate terminal punctuation is not a sentence end.

    Deterministic checks for abbreviations, initials, decimals, URLs, emails,
    dotted acronyms, and numbered-list markers.
    """
    matched = text[match_start:match_end]
    # Non-period terminals (! ? … etc.) are treated as real boundaries.
    period_rel = -1
    for i, ch in enumerate(matched):
        if ch in ".。":
            period_rel = i
            break
    if period_rel < 0:
        return False

    period_idx = match_start + period_rel

    # Decimal / version-like: digit . digit
    if period_idx > 0 and period_idx + 1 < len(text):
        if text[period_idx - 1].isdigit() and text[period_idx + 1].isdigit():
            return True

    token_start = _token_start_before(text, period_idx)
    token_end = _token_end_after(text, period_idx + 1)
    token = text[token_start:token_end]
    token_lower = token.lower()

    # URLs / emails / www hosts
    if "://" in token or "@" in token or token_lower.startswith("www."):
        return True

    line_start = text.rfind("\n", 0, period_idx) + 1

    # Numbered list marker: "1. Item" at line start OR after a prior sentence end.
    pre = text[token_start:period_idx]
    pre_lower = pre.lower()
    if _NUMBERED_LIST_PREFIX_RE.fullmatch(text[line_start:period_idx]):
        return True
    if regex.fullmatch(r"\d{1,3}", pre):
        before = text[:token_start].rstrip()
        if not before or before[-1] in ".!?…。！？\"'”’)":
            return True

    abbrevs = abbreviations_for(language)

    # Abbreviations: "Dr", "e.g", "Fig", "U.S" (as listed without final period)
    if pre_lower in abbrevs:
        return True
    # "e.g." stored as token "e.g." with pre "e.g" already; also accept dotted forms
    if pre_lower.endswith(".") and pre_lower[:-1] in abbrevs:
        return True

    # Dotted acronyms / sequences: U.S. / U.S.A. / E.U.
    if _ACRONYM_DOTTED_RE.fullmatch(pre):
        rest = text[match_end:].lstrip()
        # "U.S. economy" → not a boundary; "U.S. The" → allow boundary.
        if rest and rest[0].islower():
            return True
        # Still inside a longer dotted acronym: next char is letter then period
        if period_idx + 1 < len(text) and text[period_idx + 1].isalpha():
            return True

    # Initials: "J. Smith" / "A. B. Cook" (uppercase single letter only)
    if _SINGLE_INITIAL_RE.fullmatch(pre) and pre.isupper():
        rest = text[match_end:].lstrip()
        if rest and rest[0].isupper():
            return True

    return False


def split_sentences(text: str, language: str | None = None) -> list[tuple[int, int, str]]:
    """Split ``text`` into (start, end, content) sentence spans.

    Deterministic and language-aware: skips false boundaries from abbreviations
    (``e.g.``, ``Dr.``, ``Fig.``), initials, decimals (``3.14``), URLs, emails,
    dotted acronyms (``U.S.``), and numbered lists. Offsets refer to the
    original ``text``. Unknown languages still apply the same Unicode heuristics
    without claiming language-specific linguistic accuracy.
    """
    profile = resolve_language(language)
    spans: list[tuple[int, int, str]] = []
    start = 0
    iberian = profile.sentence_segmentation.startswith("iberian")

    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        if is_false_sentence_boundary(text, match.start(), match.end(), language):
            continue

        chunk = text[start : match.end()]
        if iberian:
            stripped = chunk.strip()
            if stripped and all(ch in _IBERIAN_OPENERS or ch.isspace() for ch in stripped):
                continue

        content = chunk.strip()
        if content:
            rel = chunk.find(content)
            abs_start = start + (rel if rel >= 0 else 0)
            spans.append((abs_start, abs_start + len(content), content))
        start = match.end()

    tail = text[start:]
    content = tail.strip()
    if content:
        rel = tail.find(content)
        abs_start = start + (rel if rel >= 0 else 0)
        spans.append((abs_start, abs_start + len(content), content))
    return spans


def list_supported_languages() -> list[dict[str, Any]]:
    """Catalog of languages with at least one non-degraded capability."""
    codes = sorted(
        {
            normalize_language_code(c) or c
            for c in set(SNOWBALL_ALGORITHMS) | set(_LANGUAGE_LEXICONS) | set(_DISPLAY_NAMES)
        }
    )
    return [resolve_language(code).to_dict() for code in codes if code and len(code) <= 3]
