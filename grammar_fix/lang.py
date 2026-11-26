"""Language detection with Apple's NaturalLanguage framework (macOS only).

Why NLLanguageRecognizer:
- built into macOS, on-device, well under 1 ms per call, no model to download;
- `languageConstraints` limits the choice to the few languages we care about,
  which makes short chat phrases far more reliable than picking from ~60;
- it returns probabilities, so the caller can tell a confident answer from a guess.

Ukrainian and Russian are among the candidates on purpose: without them Cyrillic
text would be forced into "en" or "pl". They are detected only to be skipped.

The policy (field-level vs line-level, confidence threshold, which languages are
corrected) lives in core.correct_text; this module only answers "which language?".
"""

from __future__ import annotations

from NaturalLanguage import NLLanguageRecognizer

CANDIDATES: tuple[str, ...] = ("en", "pl", "uk", "ru")


def detect(text: str, candidates: tuple[str, ...] = CANDIDATES) -> tuple[str | None, float]:
    """Return (ISO 639-1 code, probability), or (None, 0.0) when undetectable."""
    if not text.strip():
        return None, 0.0
    recognizer = NLLanguageRecognizer.alloc().init()  # cheap; one per call keeps it thread-safe
    recognizer.setLanguageConstraints_(list(candidates))
    recognizer.processString_(text)
    hypotheses = recognizer.languageHypothesesWithMaximum_(1) or {}
    if not hypotheses:
        return None, 0.0
    lang, probability = max(hypotheses.items(), key=lambda kv: kv[1])
    return str(lang), float(probability)
