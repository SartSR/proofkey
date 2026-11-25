"""Prompt for the proofreading model.

Few-shot pairs are sent as real chat turns: small local models follow
demonstrated behaviour (no explanations, keep slang/URLs, no added period)
far more reliably than instructions alone. Ollama reuses the KV cache for the
identical prefix, so the extra tokens cost almost nothing after the first call.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a proofreader for English text typed into chat messages, e-mails and forms.
Fix only spelling, grammar, punctuation and capitalization errors.
Rules:
- Reply with ONLY the corrected text. No explanations, no quotes, no prefixes.
- Keep the meaning, wording, tone and style. Do not rephrase, shorten or make it more formal.
- Keep informal words, slang and abbreviations (btw, pls, u, LGTM) as they are.
- Never change names, URLs, e-mails, @mentions, #channels, `code`, numbers or emoji.
- Do not translate. Leave non-English words as they are.
- Do not add a period at the end if the text has none.
- If the text has no errors, reply with it unchanged."""

FEW_SHOT = [
    ("i dont think we has enough time for this", "I don't think we have enough time for this"),
    ("Can you send me the report until friday? Thanks!", "Can you send me the report by Friday? Thanks!"),
    (
        "@alex pls check https://ci.example.com/job/42 when u have a sec",
        "@alex pls check https://ci.example.com/job/42 when u have a sec",
    ),
    ("Their going to review it tommorow.", "They're going to review it tomorrow."),
]


def build_messages(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in FEW_SHOT:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": text})
    return messages
