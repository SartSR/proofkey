# Proofkey

**Press ⌃⌥G in any macOS text field — Teams, Slack, Mail, your browser — and a local LLM fixes your English or Polish grammar in place.**
Minimal edits, no rewrites, nothing leaves your Mac.

```
before:  i has went to the office yesterday and forgot my laptop
after:   I went to the office yesterday and forgot my laptop

before:  Nie wiem czy zdąże na spotkanie.
after:   Nie wiem, czy zdążę na spotkanie.
```

- **Works everywhere** — any app that exposes its text field to macOS Accessibility, with a clipboard fallback for the rest.
- **English and Polish** — the language is detected per line, on-device. Text in other languages (e.g. Ukrainian) is left untouched.
- **Local and private** — talks only to [Ollama](https://ollama.com) on `localhost`.
- **Fixes, doesn't rewrite** — every model answer is validated; anything that looks like a rewrite is discarded.
- **Fast and light** — ~0.3 s per message with `gemma4:12b-mlx` (7.7 GB) on Apple Silicon.
- **Undoable** — the fix is pasted like normal input, so `⌘Z` in the app restores your text.

> Status: working MVP (v0.1). macOS only.

## How it works

1. **Hotkey** — a global ⌃⌥G registered via Carbon `RegisterEventHotKey`. No Input Monitoring permission, no keylogging, and the combo never reaches the app you type in.
2. **Read the field** — through the Accessibility API (`AXUIElement`): the selected text if there is a selection, otherwise the whole field. Electron apps (Slack, Teams, VS Code) get `AXManualAccessibility` enabled on demand. If an app exposes nothing, it falls back to `⌘C` / `⌘A ⌘C`.
3. **Split** — the text is split into lines (and long lines into sentence groups). List markers and indentation never reach the model. A line that follows a list marker or a line ending with a comma (`Dzień dobry,`) keeps the case of its first letter.
4. **Detect the language** — Apple's `NLLanguageRecognizer`, on-device and under 1 ms, limited to en/pl/uk/ru:
   - the whole field is detected first;
   - a line gets its own language only if it is long enough and the detection is confident, so short replies like "ok" follow the rest of the message;
   - lines in languages that are not enabled are skipped.
5. **Correct** — chunks go to Ollama in parallel with `temperature: 0` and the prompt and few-shot examples of their language.
6. **Validate** — an answer is rejected, and that chunk kept as-is, if it:
   - changed length too much;
   - is too dissimilar to the original;
   - lost a URL, e-mail, `@mention`, `#channel`, `` `code` `` or `:emoji:`;
   - was cut off by the token limit.
7. **Merge minimally** — only changed tokens are applied, so your quote style and apostrophes stay untouched.
8. **Paste back** — `⌘A` (whole field) + `⌘V`. Setting `AXValue` directly is ignored by React editors and breaks Undo; a paste is handled like normal input. Your clipboard is saved and restored, and the temporary content is marked transient so clipboard managers skip it.

### Safety guards

- **Password fields** (`AXSecureTextField`) are never read.
- **Rich-text fields with mentions or attachments** are not replaced as a whole. You are asked to select the plain text instead, so mentions are not destroyed.
- **Nothing is replaced if things moved on** — if you switch apps or keep typing while the model works.
- **Fields over `max_chars`** are skipped.

## Requirements

- macOS on Apple Silicon (Intel works, just slower)
- Python 3.12+
- [Ollama](https://ollama.com) running locally with at least one model pulled

## Install

```bash
git clone https://github.com/SartSR/proofkey.git && cd proofkey
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
ollama pull gemma4:12b-mlx      # or any model, see "Choosing a model"
```

## Run

```bash
.venv/bin/python -m grammar_fix
```

A ✎ icon appears in the menu bar. On first launch macOS asks for **Accessibility** permission:

1. Open System Settings → Privacy & Security → Accessibility.
2. Enable the app you launched the process from (Terminal, iTerm…).
3. Restart the process.

> ⚠️ Accessibility is granted to the *launching* app. Avoid granting it to an IDE such as VS Code: every extension would inherit the permission. Run from a dedicated terminal instead.

Then put the cursor in any text field and press **⌃⌥G**.

| Icon | Meaning |
|---|---|
| ✎ | idle |
| … | working |
| ✓ | done / no issues |
| ⚠ | see the status line in the menu |

Logs go to the terminal: one summary line per fix, with app, mode, read method, languages, chunk counts and timing.

```
17:35:18 INFO    TextEdit | whole field via AX | 55 chars | en | 1 chunk(s): 1 fixed, 0 rejected | model 0.50 s, total 1.09 s
17:36:02 INFO    Microsoft Teams | selection via AX | 74 chars | pl | 2 chunk(s): 1 fixed, 0 rejected, 1 skipped (uk) | model 0.45 s, total 0.98 s
```

To also log the text (before → after), set `GRAMMAR_FIX_DEBUG=1`.

## Configuration

`config.toml` in the project root (or the path in `$GRAMMAR_FIX_CONFIG`):

```toml
[ollama]
url = "http://127.0.0.1:11434"
model = "gemma4:12b-mlx"
timeout_s = 60
keep_alive = "30m"     # keep the model in memory between corrections

[hotkey]
combo = "ctrl+alt+g"   # modifiers: ctrl, alt, shift, cmd; key: a-z, 0-9, space

[correction]
max_chars = 8000
max_workers = 4
languages = ["en", "pl"]   # others are detected and left untouched
```

You can switch the model for the current session from the menu bar.

> 🔒 Ollama models tagged `:cloud` run on remote servers. Selecting one sends your text off the machine.

## Choosing a model

`bench/run_bench.py` runs the real pipeline (language detection + prompt + validation) over `bench/cases.jsonl` (32 English and 27 Polish cases). It reports per model and language:

- **accuracy** — output matches one of the accepted answers;
- **clean kept** — already-correct inputs left untouched;
- **untouched** — output identical to input;
- **diacr.** — Polish written without diacritics that got them added;
- **lang ok** — chunks whose detected language matches the case label;
- **rejected** — chunks discarded by the validator;
- **latency** — average and p95 time per case.

```bash
.venv/bin/python bench/run_bench.py --models gemma4:12b-mlx,qwen3.5:9b-mlx -v
.venv/bin/python bench/run_bench.py --langs pl --prompt-lang case    # model quality only, no detection
.venv/bin/python bench/run_bench.py --langs pl --prompt-lang en      # what an English-only prompt does to Polish
```

The default model with language detection:

| Language | Accuracy | Clean kept | Lang ok | Avg | p95 |
|---|---|---|---|---|---|
| English | 32/32 | 8/9 | 36/37¹ | 0.32 s | 0.49 s |
| Polish | 27/27 | 6/6 | 32/32 | 0.45 s | 0.83 s |

¹ The single "miss" is a Ukrainian line in the English set. It is correctly detected as `uk` and skipped.

Model comparison, English (32 cases):

| Model | Size | Accuracy | Clean kept | Avg | p95 |
|---|---|---|---|---|---|
| **gemma4:12b-mlx** (default) | **7.7 GB** | **32/32** | 8/9 | **0.33 s** | **0.49 s** |
| gemma4:26b-mlx (MoE) | 18 GB | 32/32 | 8/9 | 0.34 s | 0.64 s |
| qwen3.5:9b-mlx | 8.9 GB | 31/32 | 8/9 | 0.56 s | 0.99 s |
| GRMR-V3-G4B Q8_0 (grammar fine-tune) | ~4 GB | 29/32 | 7/9 | 0.54 s | 0.99 s |
| inkling-2-mini:26b-a5b-nvfp4 | 20 GB | 15/32 | 9/9 | 3.21 s | 3.88 s |

Notes:

- **MoE saves compute, not memory.** All experts stay in RAM, so `26b-a4b` needs as much memory as any 26B model.
- **Prefer models whose reasoning can be switched off.** `inkling-2-mini` keeps reasoning hidden even with `think: false` / `"none"` (~70 tokens for a one-line fix), which costs seconds and exhausts the token budget.

Add your own real-world misses to `bench/cases.jsonl`. Over time the benchmark becomes a test of *your* writing.

## Limitations (MVP)

- **English and Polish only.** Other languages are detected and skipped. Adding a language takes a prompt and few-shot examples in `grammar_fix/prompts.py`, the code in `lang.CANDIDATES`, and benchmark cases.
- **Polish diacritics get restored.** Text typed without them (`czesc, mozesz…`) comes back with them (`cześć, możesz…`). This is intended: it is a spelling fix.
- **Plain text only.** In rich-text editors (Outlook, Mail) a whole-field fix replaces formatted text with plain text. Select the plain fragment instead.
- **Hotkey-triggered, no live underlines yet.** An overlay would need per-app text-range geometry, which Electron apps report unreliably.
- **Accessibility is tied to the launching app.** A standalone `.app` bundle (py2app) would scope the permission to this tool alone; it is not built yet.

## Development

```bash
.venv/bin/pytest
```

The tests need neither Ollama nor Accessibility. They cover:

- the correction pipeline: segmentation round-trip, validation, merging, list and greeting case handling, truncated answers;
- language assignment, with a fake detector;
- real on-device language detection (macOS);
- config parsing.

| File | Role |
|---|---|
| `grammar_fix/core.py` | segmentation, language assignment, cleaning, validation, minimal merge |
| `grammar_fix/lang.py` | on-device language detection (`NLLanguageRecognizer`) |
| `grammar_fix/ollama_client.py` | stdlib HTTP client for `/api/chat` |
| `grammar_fix/prompts.py` | per-language system prompts and few-shot examples |
| `grammar_fix/text_access.py` | Accessibility read, clipboard fallback, paste back |
| `grammar_fix/app.py` | menu bar app and Carbon hotkey |
| `bench/` | model benchmark |

## Credits

The idea and parts of the correction engine come from [GrammarLLM](https://github.com/whiteh4cker-tr/grammar-llm) by whiteh4cker-tr (MIT): output cleaning, the length guard, quote-only change suppression and the token diff. GrammarLLM is a standalone editor; this project brings the same approach to every text field on macOS via Ollama.

## License

[MIT](LICENSE)
