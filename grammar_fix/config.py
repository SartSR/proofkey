"""Settings from config.toml (project root by default, or $GRAMMAR_FIX_CONFIG)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass
class Config:
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b-mlx"
    timeout_s: float = 60.0
    keep_alive: str = "30m"
    hotkey: str = "ctrl+alt+g"
    max_chars: int = 8000
    max_workers: int = 4
    languages: tuple[str, ...] = ("en", "pl")  # languages to correct; others are left untouched


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("GRAMMAR_FIX_CONFIG") or DEFAULT_PATH)
    cfg = Config()
    if not path.exists():
        return cfg
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    ollama = data.get("ollama", {})
    cfg.ollama_url = ollama.get("url", cfg.ollama_url)
    cfg.model = ollama.get("model", cfg.model)
    cfg.timeout_s = float(ollama.get("timeout_s", cfg.timeout_s))
    cfg.keep_alive = str(ollama.get("keep_alive", cfg.keep_alive))
    cfg.hotkey = data.get("hotkey", {}).get("combo", cfg.hotkey)
    corr = data.get("correction", {})
    cfg.max_chars = int(corr.get("max_chars", cfg.max_chars))
    cfg.max_workers = int(corr.get("max_workers", cfg.max_workers))
    languages = corr.get("languages", list(cfg.languages))
    if not isinstance(languages, list) or not languages or not all(isinstance(x, str) for x in languages):
        raise ValueError(f"[correction] languages must be a non-empty list of codes, got {languages!r}")
    cfg.languages = tuple(x.strip().lower() for x in languages)
    return cfg


# ------------------------------------------------------------------- hotkey

# Carbon modifier masks (Events.h) and display symbols.
_MODIFIERS = {
    "cmd": (0x0100, "⌘"),
    "command": (0x0100, "⌘"),
    "shift": (0x0200, "⇧"),
    "alt": (0x0800, "⌥"),
    "opt": (0x0800, "⌥"),
    "option": (0x0800, "⌥"),
    "ctrl": (0x1000, "⌃"),
    "control": (0x1000, "⌃"),
}
_SYMBOL_ORDER = "⌃⌥⇧⌘"

# ANSI virtual key codes (HIToolbox/Events.h). Physical positions, so they work
# with any keyboard layout, including Ukrainian.
_KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29, "o": 31,
    "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "space": 49, "/": 44, ".": 47, ",": 43, ";": 41,
}


@dataclass(frozen=True)
class Hotkey:
    keycode: int
    modifiers: int
    label: str


def parse_hotkey(combo: str) -> Hotkey:
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"Hotkey needs at least one modifier and a key: {combo!r}")
    *mods, key = parts
    if key not in _KEYCODES:
        raise ValueError(f"Unsupported key {key!r} in hotkey {combo!r}")
    mask, symbols = 0, set()
    for m in mods:
        if m not in _MODIFIERS:
            raise ValueError(f"Unknown modifier {m!r} in hotkey {combo!r}")
        bit, sym = _MODIFIERS[m]
        mask |= bit
        symbols.add(sym)
    label = "".join(s for s in _SYMBOL_ORDER if s in symbols) + key.upper()
    return Hotkey(_KEYCODES[key], mask, label)
