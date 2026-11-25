"""Correction pipeline: split text into chunks, ask the model, validate, reassemble.

Design notes (why it looks like this):
- Text from messengers/e-mail is line-oriented (lists, greetings, signatures), so
  the unit of work is a *line*. Indentation, list markers and line endings never
  reach the model and are re-attached verbatim.
- Very long lines are cut at sentence boundaries into chunks of at most
  MAX_CHUNK_CHARS, so small models do not start rewriting whole paragraphs.
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
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

Corrector = Callable[[str], str]


class ChunkRejected(Exception):
    """Raised by a corrector when its answer for one chunk is unusable
    (e.g. generation hit the token limit). Only that chunk is left unchanged."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Segment(NamedTuple):
    text: str
    needs_correction: bool
    after_marker: bool = False  # chunk follows a list/quote marker ("- ", "1. ", "> ")

MAX_CHUNK_CHARS = 600
MIN_SIMILARITY = 0.6  # char-level similarity below this = the model rewrote the text
MAX_LENGTH_DELTA_RATIO = 0.5  # reject if length changed by more than 50 %...
SHORT_TEXT_SLACK = 12  # ...or by more than this many chars for short texts
MIN_LATIN_SHARE = 0.6  # skip lines that are mostly non-Latin (e.g. Ukrainian)

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

# Whitespace after sentence-final punctuation, followed by a sentence start.
_SENTENCE_GAP_RE = re.compile(r"""(?:(?<=[.!?])|(?<=[.!?]["')\]]))\s+(?=["'(\[]?[A-Z0-9])""")

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_FENCE_RE = re.compile(r"^```[\w-]*\n(.*?)\n?```$", re.DOTALL)
_PREFIX_RE = re.compile(
    r"^(?:here(?:\s+is|'s)\s+the\s+corrected\s+(?:text|version|sentence)"
    r"|corrected(?:\s+(?:text|version|sentence))?|correction|fixed(?:\s+text)?|output)\s*:\s*",
    re.IGNORECASE,
)

# Every character is whitespace, a word char or "other", so findall() covers the string.
_TOKEN_RE = re.compile(r"\s+|\w+(?:['’]\w+)*|[^\w\s]")
_QUOTE_TABLE = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))


@dataclass
class ChunkReport:
    original: str
    result: str
    status: str  # "unchanged" | "changed" | "rejected:<reason>"


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


# --------------------------------------------------------------------------- split


def _should_correct(core: str) -> bool:
    if len(core.strip()) < 2:
        return False
    letters = [c for c in _PROTECTED_RE.sub(" ", core) if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if c.isascii())
    return latin >= len(letters) * MIN_LATIN_SHARE


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
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        prefix = _LINE_PREFIX_RE.match(body).group(1)
        rest = body[len(prefix) :]
        core = rest.rstrip()
        trailing = rest[len(core) :]
        has_marker = bool(prefix.strip())

        segs.append(Segment(prefix, False))
        if _should_correct(core):
            for n, (piece, sep) in enumerate(_split_long(core)):
                segs.append(Segment(piece, True, has_marker and n == 0))
                segs.append(Segment(sep, False))
        else:
            segs.append(Segment(core, False))
        segs.append(Segment(trailing + ending, False))
    return [s for s in segs if s.text]


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


# ---------------------------------------------------------------------- pipeline


def _keep_initial_case(original: str, fixed: str) -> str:
    """List items are often lower-case fragments; the model sees each item alone
    and capitalizes some of them, which makes the list inconsistent."""
    if original[:1].islower() and fixed[:1].isupper() and fixed[0].lower() == original[0]:
        return original[0] + fixed[1:]
    return fixed


def correct_text(text: str, corrector: Corrector, max_workers: int = 4) -> CorrectionResult:
    """Correct `text`. Corrector errors (e.g. Ollama down) propagate to the caller;
    ChunkRejected only leaves that one chunk unchanged."""
    segs = segment(text)
    idx = [i for i, s in enumerate(segs) if s.needs_correction]
    parts = [s.text for s in segs]
    reports: List[ChunkReport] = []

    def safe(chunk: str) -> Union[str, ChunkRejected]:
        try:
            return corrector(chunk)
        except ChunkRejected as e:
            return e

    if idx:
        workers = max(1, min(max_workers, len(idx)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            raws = list(pool.map(safe, [parts[i] for i in idx]))
        for i, raw in zip(idx, raws):
            original = parts[i]
            if isinstance(raw, ChunkRejected):
                fixed, status = original, "rejected:" + raw.reason
            else:
                fixed, status = postprocess(original, raw)
                if segs[i].after_marker:
                    fixed = _keep_initial_case(original, fixed)
                    if fixed == original:
                        status = "unchanged"
            parts[i] = fixed
            reports.append(ChunkReport(original, fixed, status))
    return CorrectionResult("".join(parts), reports)
