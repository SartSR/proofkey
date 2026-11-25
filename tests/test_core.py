from grammar_fix.core import (
    MAX_CHUNK_CHARS,
    ChunkRejected,
    clean_output,
    correct_text,
    minimal_merge,
    postprocess,
    segment,
    validate,
)


def identity(text: str) -> str:
    return text


def table(mapping: dict):
    """Fake model: known inputs get a fixed answer, anything else is echoed."""
    return lambda text: mapping.get(text, text)


# ------------------------------------------------------------------ segmentation


def test_segment_roundtrip_preserves_everything():
    text = "Dear John,\r\n\n  - fix the bug\n1. deploy it  \n> quoted line\nПривіт!\n\tend"
    assert "".join(s.text for s in segment(text)) == text


def test_identity_model_changes_nothing():
    text = "Hello there.\n\n- item one\n- item two\nBest,\nAnna\n"
    assert correct_text(text, identity).text == text


def test_list_markers_and_indent_never_reach_the_model():
    seen = []

    def spy(t):
        seen.append(t)
        return t

    correct_text("  - first item\n2) second item\n> quoted", spy)
    assert sorted(seen) == ["first item", "quoted", "second item"]  # chunks run in parallel


def test_non_latin_lines_are_skipped():
    seen = []
    correct_text("Привіт, можеш глянути PR?\nok", lambda t: seen.append(t) or t)
    assert seen == ["ok"]


def test_long_line_is_split_at_sentences_and_reassembled():
    sentence = "This sentence is long enough to matter for chunking purposes. "
    text = (sentence * (MAX_CHUNK_CHARS // len(sentence) * 3)).rstrip()
    chunks = [s.text for s in segment(text) if s.needs_correction]
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    assert correct_text(text, identity).text == text


# ---------------------------------------------------------------------- pipeline


def test_multiline_email_fixes_only_the_broken_line():
    text = "Dear John,\nI wanted to follow up on our discusion.\n\nBest regards,\nAnna"
    model = table({"I wanted to follow up on our discusion.": "I wanted to follow up on our discussion."})
    result = correct_text(text, model)
    assert result.text == text.replace("discusion", "discussion")
    assert result.changed_count == 1


def test_rewrite_is_rejected():
    original = "can u check the logs pls"
    model = table({original: "Would you kindly review the application logs at your earliest convenience?"})
    result = correct_text(original, model)
    assert result.text == original
    assert result.chunks[0].status.startswith("rejected")


def test_lost_url_or_mention_is_rejected():
    original = "@maria see https://jira.example.com/browse/OPS-1 its broken"
    model = table({original: "Maria, see https://jira.example.com/browse/OPS-1. It's broken"})
    assert correct_text(original, model).text == original


def test_url_trailing_period_is_not_part_of_protected_token():
    original = "its at https://example.com/a."
    assert validate(original, "It's at https://example.com/a.") is None


def test_list_items_keep_initial_case():
    text = "- update the docs\n- fix the bug wich affects users\n* deploy it"
    model = table({
        "update the docs": "Update the docs",
        "fix the bug wich affects users": "Fix the bug which affects users",
        "deploy it": "Deploy it",
    })
    result = correct_text(text, model)
    assert result.text == "- update the docs\n- fix the bug which affects users\n* deploy it"
    assert sorted(c.status for c in result.chunks) == ["changed", "unchanged", "unchanged"]


def test_plain_lines_still_get_capitalized():
    model = table({"thanks for you help": "Thanks for your help"})
    assert correct_text("thanks for you help", model).text == "Thanks for your help"


def test_rejected_chunk_is_left_unchanged_others_still_fixed():
    def model(chunk):
        if chunk.startswith("The meeting"):
            raise ChunkRejected("truncated")
        return chunk.replace("wich", "which")

    text = "The meeting have been moved to 3 PM.\nthe bug wich we saw"
    result = correct_text(text, model)
    assert result.text == "The meeting have been moved to 3 PM.\nthe bug which we saw"
    assert sorted(c.status for c in result.chunks) == ["changed", "rejected:truncated"]


def test_errors_from_model_propagate():
    def broken(_):
        raise RuntimeError("ollama down")

    try:
        correct_text("some text here", broken)
    except RuntimeError as e:
        assert "ollama down" in str(e)
    else:
        raise AssertionError("expected the error to propagate")


# ---------------------------------------------------------------- post-process


def test_clean_output_strips_wrapping():
    original = "i has a question"
    assert clean_output("<think>hmm</think>\nCorrected: I have a question", original) == "I have a question"
    assert clean_output('"I have a question"', original) == "I have a question"
    assert clean_output("```\nI have a question\n```", original) == "I have a question"
    assert clean_output("I have a question\n\nI fixed the verb.", original) == "I have a question"


def test_added_final_period_is_dropped_but_kept_when_original_had_one():
    assert postprocess("thanks for you help", "Thanks for your help.") == ("Thanks for your help", "changed")
    assert postprocess("Thanks for you help.", "Thanks for your help.")[0] == "Thanks for your help."


def test_quote_style_is_preserved():
    original = "It’s “fine”, isnt it?"
    fixed = minimal_merge(original, "It's \"fine\", isn't it?")
    assert fixed == "It’s “fine”, isn't it?"


def test_unchanged_answer_reports_unchanged():
    assert postprocess("All good here.", "All good here.") == ("All good here.", "unchanged")
