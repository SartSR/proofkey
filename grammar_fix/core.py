"""Correction pipeline: split text into chunks, ask the model, validate, reassemble.

Design notes (why it looks like this):
- Text from messengers/e-mail is line-oriented (lists, greetings, signatures), so
  the unit of work is a *line*. Indentation, list markers and line endings never
  reach the model and are re-attached verbatim.
- Very long lines are cut at sentence boundaries into chunks of at most
  MAX_CHUNK_CHARS, so small models do not start rewriting whole paragraphs.
- Each chunk gets a language. The detector is injected (see lang.py), so this
  module stays free of any OS dependency and is testable with fakes. Chunks in
  languages that are not enabled (e.g. Ukrainian) are left untouched.
- Every model answer is validated. Anything that looks like a rewrite (length
  jump, low similarity, lost URL/@mention/`code`) is dropped and the original
  chunk is kept. A missed fix is cheap; a silently mangled message is not.
- The accepted answer is merged back token by token: unchanged tokens keep the
  user's original characters (quote style, apostrophes), so only real edits land.

Ported ideas (MIT) from https://github.com/whiteh4cker-tr/grammar-llm (python branch):
output cleaning, length guard, quote-only change suppression, token diff.
"""

from __future__ import annotations

import difflib
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

# corrector(chunk, lang) -> corrected chunk
Corrector = Callable[[str, str], str]
# detect(text) -> (language code or None, probability)
Detector = Callable[[str], Tuple[Optional[str], float]]


class ChunkRejected(Exception):
    """Raised by a corrector when its answer for one chunk is unusable
    (e.g. generation hit the token limit). Only that chunk is left unchanged."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Segment(NamedTuple):
    text: str
    needs_correction: bool
    # Keep the first letter's case as typed: the chunk follows a list/quote marker
    # ("- ", "1. ", "> ") or continues a line that ended with a comma
    # ("Dzień dobry,\nchciałbym…"). The model sees the chunk alone and would
    # otherwise capitalise it as a sentence start.
    keep_initial_case: bool = False


MAX_CHUNK_CHARS = 600
MIN_SIMILARITY = 0.6  # char-level similarity below this = the model rewrote the text
MAX_LENGTH_DELTA_RATIO = 0.5  # reject if length changed by more than 50 %...
SHORT_TEXT_SLACK = 12  # ...or by more than this many chars for short texts
MIN_LATIN_SHARE = 0.6  # without a detector: skip lines that are mostly non-Latin
MIN_CHARS_FOR_OWN_LANGUAGE = 20  # shorter chunks inherit the language of the whole field
MIN_LANGUAGE_CONFIDENCE = 0.6  # below this a chunk's own detection is not trusted

# Leading indentation plus an optional list / quote marker: "  - ", "1. ", "> ".
_LINE_PREFIX_RE = re.compile(r"^(\s*(?:(?:[-*•‣◦>]|\d{1,3}[.)])\s+)?)")

# Tokens the model must never touch. No capturing groups: findall() returns full matches.
_PROTECTED_RE = re.compile(
    r"""
      https?://[^\s<>"]+?(?=[.,!?;:)\]]*(?:\s|$))   # URL, without trailing punctuation
    | www\.[^\s<>"]+?(?=[.,!?;:)\]]*(?:\s|$))
    | [\w.+-]+@[\w-]+(?:\.[\w-]+)+                   # e-mail
    | (?<![\w@])[@#]\w(?:[\w.-]*\w)?                 # @mention, #channel
    | `[^`\n]+`                                      # inline code
    | :[a-z_+-][a-z0-9_+-]*:                         # :emoji:
    """,
    re.VERBOSE,
)

# Whitespace after sentence-final punctuation, followed by a sentence start
# (Latin incl. Polish capitals, or a digit).
_SENTENCE_GAP_RE = re.compile(
    r"""(?:(?<=[.!?])|(?<=[.!?]["')\]]))\s+(?=["'(\[„]?[A-ZĄĆĘŁŃÓŚŹŻ0-9])"""
)

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_FENCE_RE = re.compile(r"^```[\w-]*\n(.*?)\n?```$", re.DOTALL)
_PREFIX_RE = re.compile(
    r"^(?:here(?:\s+is|'s)\s+the\s+corrected\s+(?:text|version|sentence)"
    r"|corrected(?:\s+(?:text|version|sentence))?|correction|fixed(?:\s+text)?|output"
    # Polish: "Poprawiony tekst:", "Oto poprawiona wersja:", "Poprawka:"
    r"|(?:oto\s+)?poprawion[ya](?:\s+(?:tekst|wersja|zdanie))?|poprawka)\s*:\s*",
    re.IGNORECASE,
)

# Every character is whitespace, a word char or "other", so findall() covers the string.
_TOKEN_RE = re.compile(r"\s+|\w+(?:['’]\w+)*|[^\w\s]")
_QUOTE_TABLE = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("„", "”"))


@dataclass
class ChunkReport:
    original: str
    result: str
    status: str  # "unchanged" | "changed" | "rejected:<reason>" | "skipped:<lang>"
    lang: Optional[str] = None


@dataclass
class CorrectionResult:
    text: str
    chunks: List[ChunkReport]

    @property
    def changed_count(self) -> int:
        return sum(1 for c in self.chunks if c.status == "changed")

    @property
    def rejected_count(self) -> int:
        return sum(1 for c in self.chunks if c.status.startswith("rejected"))

    @property
    def skipped_count(self) -> int:
        return sum(1 for c in self.chunks if c.status.startswith("skipped"))

    @property
    def languages(self) -> List[str]:
        """Languages of the chunks sent to the model, in order of first use."""
        seen: List[str] = []
        for c in self.chunks:
            if c.lang and not c.status.startswith("skipped") and c.lang not in seen:
                seen.append(c.lang)
        return seen


# --------------------------------------------------------------------------- split


def _has_text(core: str) -> bool:
    """Worth sending to a model at all: some letters outside URLs/mentions/code."""
    if len(core.strip()) < 2:
        return False
    return any(c.isalpha() for c in _PROTECTED_RE.sub(" ", core))


def _mostly_latin(core: str) -> bool:
    letters = [c for c in _PROTECTED_RE.sub(" ", core) if c.isalpha()]
    latin = sum(1 for c in letters if c.isascii() or "À" <= c <= "ɏ")  # incl. Latin Extended
    return bool(letters) and latin >= len(letters) * MIN_LATIN_SHARE


def _split_long(core: str) -> List[Tuple[str, str]]:
    """Split a long line into (chunk, separator) pairs; concatenation == core."""
    if len(core) <= MAX_CHUNK_CHARS:
        return [(core, "")]
    sentences: List[Tuple[str, str]] = []
    pos = 0
    for m in _SENTENCE_GAP_RE.finditer(core):
        sentences.append((core[pos : m.start()], m.group()))
        pos = m.end()
    sentences.append((core[pos:], ""))

    out: List[Tuple[str, str]] = []
    cur, cur_sep = sentences[0]
    for sent, sep in sentences[1:]:
        if len(cur) + len(cur_sep) + len(sent) <= MAX_CHUNK_CHARS:
            cur, cur_sep = cur + cur_sep + sent, sep
        else:
            out.append((cur, cur_sep))
            cur, cur_sep = sent, sep
    out.append((cur, cur_sep))
    return out


def segment(text: str) -> List[Segment]:
    """Cut text into segments; "".join(s.text for s in segments) == text."""
    segs: List[Segment] = []
    prev_line = ""  # last non-blank line, to detect "Dzień dobry," continuations
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        prefix = _LINE_PREFIX_RE.match(body).group(1)
        rest = body[len(prefix) :]
        core = rest.rstrip()
        trailing = rest[len(core) :]
        keep_case = bool(prefix.strip()) or prev_line.endswith(",")

        segs.append(Segment(prefix, False))
        if _has_text(core):
            for n, (piece, sep) in enumerate(_split_long(core)):
                segs.append(Segment(piece, True, keep_case and n == 0))
                segs.append(Segment(sep, False))
        else:
            segs.append(Segment(core, False))
        segs.append(Segment(trailing + ending, False))
        if body.strip():
            prev_line = body.rstrip()
    return [s for s in segs if s.text]


# ---------------------------------------------------------------------- languages


def assign_languages(
    text: str, chunks: Sequence[str], detect: Optional[Detector], languages: Sequence[str]
) -> List[Optional[str]]:
    """Language for every chunk; None = do not correct.

    Without a detector every Latin-script chunk gets languages[0]. With one, the
    whole field is detected first (long text = reliable), and a chunk overrides
    it only when it is long enough and its own detection is confident. Short
    replies like "ok" or "dzięki" therefore follow the rest of the message.
    """
    if detect is None:
        return [languages[0] if _mostly_latin(c) else None for c in chunks]
    field_lang, _ = detect(text)
    out: List[Optional[str]] = []
    for chunk in chunks:
        lang = field_lang
        if len(chunk) >= MIN_CHARS_FOR_OWN_LANGUAGE:
            own, probability = detect(chunk)
            if own and probability >= MIN_LANGUAGE_CONFIDENCE:
                lang = own
        out.append(lang)
    return out


# ------------------------------------------------------------------ post-process


def clean_output(raw: str, original: str) -> str:
    """Strip typical LLM wrapping: think blocks, special tokens, fences, prefixes, quotes."""
    s = _THINK_RE.sub("", raw or "")
    s = _SPECIAL_TOKEN_RE.sub("", s).strip()
    fence = _FENCE_RE.match(s)
    if fence:
        s = fence.group(1).strip()
    s = s.split("\n\n")[0].strip()  # drop any explanation after a blank line
    if not _PREFIX_RE.match(original):
        s = _PREFIX_RE.sub("", s, count=1)
    for left, right in _QUOTE_PAIRS:
        if (
            len(s) >= 2
            and s.startswith(left)
            and s.endswith(right)
            and not (original.startswith(left) and original.endswith(right))
        ):
            s = s[1:-1].strip()
    return s


def _drop_added_final_period(original: str, corrected: str) -> str:
    """Chat messages often have no final period; do not add one."""
    if (
        corrected.endswith(".")
        and not corrected.endswith("..")
        and not original.rstrip().endswith((".", "!", "?", "…"))
    ):
        return corrected[:-1]
    return corrected


def validate(original: str, corrected: str) -> Optional[str]:
    """Return a rejection reason, or None if the correction looks safe."""
    if not corrected.strip():
        return "empty"
    if "\n" in corrected:
        return "newline"
    delta = abs(len(corrected) - len(original))
    if delta > max(SHORT_TEXT_SLACK, len(original) * MAX_LENGTH_DELTA_RATIO):
        return "length"
    ratio = difflib.SequenceMatcher(None, original.lower(), corrected.lower(), autojunk=False).ratio()
    if ratio < MIN_SIMILARITY:
        return "rewrite"
    for token in set(_PROTECTED_RE.findall(original)):
        if token not in corrected:
            return "protected"
    return None


def minimal_merge(original: str, corrected: str) -> str:
    """Apply only the real edits; unchanged tokens keep the original characters."""
    a = _TOKEN_RE.findall(original)
    b = _TOKEN_RE.findall(corrected)
    matcher = difflib.SequenceMatcher(
        None, [t.translate(_QUOTE_TABLE) for t in a], [t.translate(_QUOTE_TABLE) for t in b], autojunk=False
    )
    out: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        out.extend(a[i1:i2] if tag == "equal" else b[j1:j2])
    return "".join(out)


def postprocess(original: str, raw: str) -> Tuple[str, str]:
    candidate = _drop_added_final_period(original, clean_output(raw, original))
    reason = validate(original, candidate)
    if reason:
        return original, "rejected:" + reason
    merged = minimal_merge(original, candidate)
    return merged, ("changed" if merged != original else "unchanged")


def _keep_initial_case(original: str, fixed: str) -> str:
    if original[:1].islower() and fixed[:1].isupper() and fixed[0].lower() == original[0]:
        return original[0] + fixed[1:]
    return fixed


# ---------------------------------------------------------------------- pipeline


def correct_text(
    text: str,
    corrector: Corrector,
    max_workers: int = 4,
    detect: Optional[Detector] = None,
    languages: Sequence[str] = ("en",),
) -> CorrectionResult:
    """Correct `text`. Corrector errors (e.g. Ollama down) propagate to the caller;
    ChunkRejected only leaves that one chunk unchanged."""
    segs = segment(text)
    parts = [s.text for s in segs]
    candidates = [i for i, s in enumerate(segs) if s.needs_correction]
    langs = assign_languages(text, [parts[i] for i in candidates], detect, languages)

    reports: dict = {}
    jobs: List[Tuple[int, str]] = []
    for i, lang in zip(candidates, langs):
        if lang in languages:
            jobs.append((i, lang))
        else:
            reports[i] = ChunkReport(parts[i], parts[i], f"skipped:{lang or 'unknown'}", lang)

    def safe(job: Tuple[int, str]) -> Union[str, ChunkRejected]:
        i, lang = job
        try:
            return corrector(parts[i], lang)
        except ChunkRejected as e:
            return e

    if jobs:
        workers = max(1, min(max_workers, len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            raws = list(pool.map(safe, jobs))
        for (i, lang), raw in zip(jobs, raws):
            original = parts[i]
            if isinstance(raw, ChunkRejected):
                fixed, status = original, "rejected:" + raw.reason
            else:
                fixed, status = postprocess(original, raw)
                if segs[i].keep_initial_case:
                    fixed = _keep_initial_case(original, fixed)
                    if fixed == original:
                        status = "unchanged"
            parts[i] = fixed
            reports[i] = ChunkReport(original, fixed, status, lang)

    return CorrectionResult("".join(parts), [reports[i] for i in sorted(reports)])
