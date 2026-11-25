# Proofkey

**Press ⌃⌥G in any macOS text field — Teams, Slack, Mail, your browser — and a local LLM fixes your English grammar in place.**
Minimal edits, no rewrites, nothing leaves your Mac.

```
before:  i has went to the office yesterday and forgot my laptop
after:   I went to the office yesterday and forgot my laptop
```

- **Works everywhere** — any app that exposes its text field to macOS Accessibility, with a clipboard fallback for the rest.
- **Local and private** — talks only to [Ollama](https://ollama.com) on `localhost`.
- **Fixes, doesn't rewrite** — every model answer is validated; anything that looks like a rewrite is discarded.
- **Fast** — ~0.4 s per message with `gemma4:26b-mlx` on Apple Silicon.
- **Undoable** — the fix is pasted like normal input, so `⌘Z` in the app restores your text.

> Status: working MVP (v0.1). macOS only.

## How it works

1. **Hotkey** — a global ⌃⌥G registered via Carbon `RegisterEventHotKey`. No Input Monitoring permission, no keylogging, and the combo never reaches the app you type in.
2. **Read the field** — through the Accessibility API (`AXUIElement`): the selected text if there is a selection, otherwise the whole field. Electron apps (Slack, Teams, VS Code) get `AXManualAccessibility` enabled on demand. If an app exposes nothing, it falls back to `⌘C` / `⌘A ⌘C`.
3. **Correct** — the text is split into lines (and long lines into sentence groups). List markers and indentation never reach the model. Chunks go to Ollama in parallel with `temperature: 0`, a strict system prompt and few-shot examples.
4. **Validate** — an answer is rejected, and that chunk kept as-is, if it:
   - changed length too much;
   - is too dissimilar to the original;
   - lost a URL, e-mail, `@mention`, `#channel`, `` `code` `` or `:emoji:`;
   - was cut off by the token limit.
5. **Merge minimally** — only changed tokens are applied, so your quote style and apostrophes stay untouched.
6. **Paste back** — `⌘A` (whole field) + `⌘V`. Setting `AXValue` directly is ignored by React editors and breaks Undo; a paste is handled like normal input. Your clipboard is saved and restored, and the temporary content is marked transient so clipboard managers skip it.

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
ollama pull gemma4:26b-mlx      # or any model, see "Choosing a model"
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

Logs go to the terminal: one summary line per fix, with app, mode, read method, chunk counts and timing.

```
17:35:18 INFO    TextEdit | whole field via AX | 55 chars | 1 chunk(s): 1 fixed, 0 rejected | model 0.50 s, total 1.09 s
```

To also log the text (before → after), set `GRAMMAR_FIX_DEBUG=1`.

## Configuration

`config.toml` in the project root (or the path in `$GRAMMAR_FIX_CONFIG`):

```toml
[ollama]
url = "http://127.0.0.1:11434"
model = "gemma4:26b-mlx"
timeout_s = 60
keep_alive = "30m"     # keep the model in memory between corrections

[hotkey]
combo = "ctrl+alt+g"   # modifiers: ctrl, alt, shift, cmd; key: a-z, 0-9, space

[correction]
max_chars = 8000
max_workers = 4
```

You can switch the model for the current session from the menu bar.

> 🔒 Ollama models tagged `:cloud` run on remote servers. Selecting one sends your text off the machine.

## Choosing a model

`bench/run_bench.py` runs the real pipeline (prompt + validation) over `bench/cases.jsonl` and reports four metrics:

- **accuracy** — output matches one of the accepted answers;
- **clean kept** — already-correct inputs left untouched;
- **rejected** — chunks discarded by the validator;
- **latency** — time per case.

```bash
.venv/bin/python bench/run_bench.py --models gemma4:26b-mlx,qwen3.5:9b-mlx -v
```

Results on an Apple Silicon Mac (32 cases):

| Model | Accuracy | Clean kept | Avg | p95 |
|---|---|---|---|---|
| **gemma4:26b-mlx** | **32/32** | 8/9 | **0.37 s** | 0.68 s |
| qwen3.5:9b-mlx | 30/32 | 8/9 | 0.57 s | 1.0 s |
| GRMR-V3-G4B Q8_0 (grammar fine-tune) | 29/32 | 7/9 | 0.54 s | 0.99 s |

Add your own real-world misses to `bench/cases.jsonl`. Over time the benchmark becomes a test of *your* writing.

## Limitations (MVP)

- **English only.** Lines that are mostly non-Latin are skipped on purpose.
- **Plain text only.** In rich-text editors (Outlook, Mail) a whole-field fix replaces formatted text with plain text. Select the plain fragment instead.
- **Hotkey-triggered, no live underlines yet.** An overlay would need per-app text-range geometry, which Electron apps report unreliably.
- **Accessibility is tied to the launching app.** A standalone `.app` bundle (py2app) would scope the permission to this tool alone; it is not built yet.

## Development

```bash
.venv/bin/pytest
```

The tests cover the correction pipeline (segmentation round-trip, validation, merging, list handling, truncated answers) and config parsing. They need neither Ollama nor Accessibility.

| File | Role |
|---|---|
| `grammar_fix/core.py` | segmentation, cleaning, validation, minimal merge |
| `grammar_fix/ollama_client.py` | stdlib HTTP client for `/api/chat` |
| `grammar_fix/prompts.py` | system prompt and few-shot examples |
| `grammar_fix/text_access.py` | Accessibility read, clipboard fallback, paste back |
| `grammar_fix/app.py` | menu bar app and Carbon hotkey |
| `bench/` | model benchmark |

## Credits

The idea and parts of the correction engine come from [GrammarLLM](https://github.com/whiteh4cker-tr/grammar-llm) by whiteh4cker-tr (MIT): output cleaning, the length guard, quote-only change suppression and the token diff. GrammarLLM is a standalone editor; this project brings the same approach to every text field on macOS via Ollama.

## License

[MIT](LICENSE)
