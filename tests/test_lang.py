"""Real NLLanguageRecognizer checks (macOS only; skipped elsewhere)."""

import pytest

pytest.importorskip("NaturalLanguage")

from grammar_fix.lang import detect  # noqa: E402


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I wanted to follow up on our discussion from last week.", "en"),
        ("Chciałbym zapytać, czy raport jest już gotowy.", "pl"),
        ("Spotkanie zostało przełożone na poniedziałek.", "pl"),
        ("Привіт, можеш сьогодні глянути цей пул-реквест?", "uk"),
    ],
)
def test_detects_full_sentences(text, expected):
    lang, probability = detect(text)
    assert lang == expected
    assert probability > 0.5


def test_polish_without_diacritics_is_still_polish():
    lang, _ = detect("czesc, mozesz mi wyslac ten plik z raportem?")
    assert lang == "pl"


def test_empty_text_is_undetectable():
    assert detect("   ") == (None, 0.0)
