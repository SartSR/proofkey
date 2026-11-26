"""Prompts for the proofreading model, one per language.

Few-shot pairs are sent as real chat turns: small local models follow
demonstrated behaviour (no explanations, keep slang/URLs, no added period)
far more reliably than instructions alone. Ollama reuses the KV cache for the
identical prefix, so the extra tokens cost almost nothing after the first call.
Each language has its own prefix, so each one is cached separately.
"""

from __future__ import annotations

SYSTEM_PROMPT_EN = """You are a proofreader for English text typed into chat messages, e-mails and forms.
Fix only spelling, grammar, punctuation and capitalization errors.
Rules:
- Reply with ONLY the corrected text. No explanations, no quotes, no prefixes.
- Keep the meaning, wording, tone and style. Do not rephrase, shorten or make it more formal.
- Keep informal words, slang and abbreviations (btw, pls, u, LGTM) as they are.
- Never change names, URLs, e-mails, @mentions, #channels, `code`, numbers or emoji.
- Do not translate. Leave non-English words as they are.
- Do not add a period at the end if the text has none.
- If the text has no errors, reply with it unchanged."""

FEW_SHOT_EN = [
    ("i dont think we has enough time for this", "I don't think we have enough time for this"),
    ("Can you send me the report until friday? Thanks!", "Can you send me the report by Friday? Thanks!"),
    (
        "@alex pls check https://ci.example.com/job/42 when u have a sec",
        "@alex pls check https://ci.example.com/job/42 when u have a sec",
    ),
    ("Their going to review it tommorow.", "They're going to review it tomorrow."),
]

# Instructions stay in English (models follow English instructions most
# reliably); the examples are Polish so the model imitates Polish corrections.
# Diacritics policy is deliberately NOT stated yet: the benchmark measures what
# the model does on its own before we decide.
SYSTEM_PROMPT_PL = """You are a proofreader for Polish text typed into chat messages, e-mails and forms.
Fix only spelling, grammar, punctuation (including mandatory commas) and capitalization errors.
Reply in Polish.
Rules:
- Reply with ONLY the corrected text. No explanations, no quotes, no prefixes.
- Keep the meaning, wording, tone and style. Do not rephrase, shorten or make it more formal.
- Keep informal words, slang and abbreviations (spoko, nwm, btw) as they are.
- Never change names, URLs, e-mails, @mentions, #channels, `code`, numbers or emoji.
- Do not translate. Keep English words and phrases as they are.
- Do not add a period at the end if the text has none.
- If the text has no errors, reply with it unchanged."""

FEW_SHOT_PL = [
    ("Wczoraj poszłem do sklepu i kupiłem chleb", "Wczoraj poszedłem do sklepu i kupiłem chleb"),
    (
        "Myśle że to dobry pomysł, ale musimy to przedyskutować.",
        "Myślę, że to dobry pomysł, ale musimy to przedyskutować.",
    ),
    (
        "@ania spoko, zerknij na https://ci.example.com/job/42 jak znajdziesz chwilę",
        "@ania spoko, zerknij na https://ci.example.com/job/42, jak znajdziesz chwilę",
    ),
    ("Wogóle nie wiem o co chodzi w tym raporcie.", "W ogóle nie wiem, o co chodzi w tym raporcie."),
]

PROMPTS = {
    "en": (SYSTEM_PROMPT_EN, FEW_SHOT_EN),
    "pl": (SYSTEM_PROMPT_PL, FEW_SHOT_PL),
}
LANGUAGES = tuple(PROMPTS)

# Backwards-compatible names.
SYSTEM_PROMPT = SYSTEM_PROMPT_EN
FEW_SHOT = FEW_SHOT_EN


def build_messages(text: str, lang: str = "en") -> list[dict]:
    if lang not in PROMPTS:
        raise ValueError(f"No prompt for language {lang!r}; available: {', '.join(LANGUAGES)}")
    system, few_shot = PROMPTS[lang]
    messages = [{"role": "system", "content": system}]
    for user, assistant in few_shot:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": text})
    return messages
