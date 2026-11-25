"""Menu bar app: global hotkey -> grab focused field -> Ollama -> paste back.

Hotkey: Carbon RegisterEventHotKey via ctypes. Chosen over pynput / event taps
because it needs no Input Monitoring permission, does not observe every
keystroke system-wide, and swallows the combo so it never reaches the target app.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from collections.abc import Callable

import rumps
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSBeep
from PyObjCTools import AppHelper

from . import text_access
from .config import Config, Hotkey, load_config, parse_hotkey
from .core import correct_text
from .ollama_client import OllamaClient, OllamaError
from .text_access import TextAccessError

log = logging.getLogger("grammar_fix")

ICON_IDLE, ICON_BUSY, ICON_OK, ICON_WARN = "✎", "…", "✓", "⚠"
RESULT_ICON_SECONDS = 2.5


# ------------------------------------------------------------- Carbon hotkey

_carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HandlerProc = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)

_carbon.GetEventDispatcherTarget.restype = ctypes.c_void_p
_carbon.InstallEventHandler.argtypes = [
    ctypes.c_void_p, _HandlerProc, ctypes.c_uint32, ctypes.POINTER(_EventTypeSpec),
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
]
_carbon.InstallEventHandler.restype = ctypes.c_int32
_carbon.RegisterEventHotKey.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32, _EventHotKeyID, ctypes.c_void_p,
    ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
]
_carbon.RegisterEventHotKey.restype = ctypes.c_int32


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


class GlobalHotKey:
    def __init__(self, hotkey: Hotkey, callback: Callable[[], None]):
        def handler(_call_ref, _event, _user_data):
            try:
                callback()
            except Exception:  # never let an exception cross the C boundary
                log.exception("Hotkey callback failed")
            return 0

        self._handler = _HandlerProc(handler)  # keep a reference: C holds the pointer
        target = _carbon.GetEventDispatcherTarget()
        spec = _EventTypeSpec(_fourcc("keyb"), 5)  # kEventClassKeyboard, kEventHotKeyPressed
        self._handler_ref = ctypes.c_void_p()
        status = _carbon.InstallEventHandler(
            target, self._handler, 1, ctypes.byref(spec), None, ctypes.byref(self._handler_ref)
        )
        if status != 0:
            raise RuntimeError(f"InstallEventHandler failed: {status}")
        self._hotkey_ref = ctypes.c_void_p()
        status = _carbon.RegisterEventHotKey(
            hotkey.keycode, hotkey.modifiers, _EventHotKeyID(_fourcc("GFIX"), 1),
            target, 0, ctypes.byref(self._hotkey_ref),
        )
        if status != 0:
            raise RuntimeError(f"Hotkey {hotkey.label} is taken by another app (status {status})")


# ------------------------------------------------------------------ the app


class GrammarFixApp(rumps.App):
    def __init__(self, cfg: Config):
        super().__init__("Grammar Fix", title=ICON_IDLE, quit_button=None)
        self.cfg = cfg
        self.hotkey = parse_hotkey(cfg.hotkey)
        self.client = OllamaClient(cfg.ollama_url, cfg.model, cfg.timeout_s, cfg.keep_alive)
        self._busy = threading.Lock()

        self.status_item = rumps.MenuItem("Starting…")
        self.model_menu = rumps.MenuItem("Model")
        self.model_menu.add(rumps.MenuItem(cfg.model))
        self.menu = [
            rumps.MenuItem(f"Fix text   {self.hotkey.label}", callback=lambda _: self.trigger()),
            self.model_menu,
            None,
            self.status_item,
            rumps.MenuItem("Accessibility permission…", callback=lambda _: self._check_permission(True)),
            None,
            rumps.MenuItem("Quit", callback=lambda _: rumps.quit_application()),
        ]
        self._hotkey = GlobalHotKey(self.hotkey, self.trigger)
        log.info("Hotkey %s registered", self.hotkey.label)
        threading.Thread(target=self._startup, daemon=True).start()

    # ---------------------------------------------------------------- ui
    def _ui(self, fn: Callable, *args) -> None:
        AppHelper.callAfter(fn, *args)  # AppKit must be touched on the main thread

    def _set_status(self, icon: str, message: str, reset: bool = False) -> None:
        def update():
            self.title = icon
            self.status_item.title = message
            if reset:
                AppHelper.callLater(RESULT_ICON_SECONDS, self._reset_icon)

        self._ui(update)

    def _reset_icon(self) -> None:
        if not self._busy.locked():
            self.title = ICON_IDLE

    def _check_permission(self, prompt: bool) -> bool:
        ok = text_access.is_trusted(prompt=prompt)
        if not ok:
            self._set_status(ICON_WARN, "Grant Accessibility: System Settings → Privacy & Security")
        return ok

    # ----------------------------------------------------------- startup
    def _startup(self) -> None:
        if self._check_permission(prompt=True):
            log.info("Accessibility: granted")
        else:
            log.error(
                "Accessibility: NOT granted. Enable the app that runs this process "
                "(Terminal, iTerm...) in System Settings → Privacy & Security → Accessibility, then restart"
            )
        try:
            models = self.client.list_models()
        except OllamaError as e:
            log.error("%s", e)
            self._set_status(ICON_WARN, str(e))
            return
        log.info("Ollama %s: %d models available", self.cfg.ollama_url, len(models))
        self._ui(self._fill_models, models)
        if self.cfg.model not in models:
            log.error("Model %s not found in Ollama. Available: %s", self.cfg.model, ", ".join(models))
            self._set_status(ICON_WARN, f"Model {self.cfg.model} not found in Ollama")
            return
        self._warm_up(self.cfg.model)

    def _warm_up(self, model: str) -> None:
        self._set_status(ICON_BUSY, f"Loading {model}…")
        log.info("Loading %s…", model)
        started = time.perf_counter()
        try:
            self.client.warm_up()
        except OllamaError as e:
            log.error("Loading %s failed: %s", model, e)
            self._set_status(ICON_WARN, str(e))
            return
        log.info("Ready: %s (loaded in %.1f s). Press %s in any text field", model,
                 time.perf_counter() - started, self.hotkey.label)
        self._set_status(ICON_IDLE, f"Ready: {model}")

    def _fill_models(self, models: list[str]) -> None:
        self.model_menu.clear()
        for name in models:
            item = rumps.MenuItem(name, callback=self._select_model)
            item.state = int(name == self.client.model)
            self.model_menu.add(item)

    def _select_model(self, item: rumps.MenuItem) -> None:
        log.info("Model switched to %s (this session only; default is in config.toml)", item.title)
        self.client.model = item.title
        for other in self.model_menu.values():
            other.state = int(other.title == item.title)
        threading.Thread(target=self._warm_up, args=(item.title,), daemon=True).start()

    # ------------------------------------------------------------ action
    def trigger(self) -> None:
        if not self._busy.acquire(blocking=False):
            return  # a correction is already running
        self.title = ICON_BUSY
        threading.Thread(target=self._run_fix, daemon=True).start()

    def _run_fix(self) -> None:
        started = time.perf_counter()
        where = "?"
        try:
            target = text_access.capture()
            where = f"{target.app_name} | {target.mode} via {target.via} | {len(target.text)} chars"
            if len(target.text) > self.cfg.max_chars:
                raise TextAccessError(f"Text is longer than {self.cfg.max_chars} chars: skipped")

            model_started = time.perf_counter()
            result = correct_text(target.text, self.client.correct, self.cfg.max_workers)
            model_s = time.perf_counter() - model_started
            for chunk in result.chunks:
                if chunk.status.startswith("rejected"):
                    log.warning("  chunk %s (left unchanged)", chunk.status)
                # Text goes to the log only in debug mode (GRAMMAR_FIX_DEBUG=1).
                log.debug("  %s: %r -> %r", chunk.status, chunk.original, chunk.result)
            counts = (f"{len(result.chunks)} chunk(s): {result.changed_count} fixed, "
                      f"{result.rejected_count} rejected")

            if result.text == target.text:
                log.info("%s | %s | no issues | model %.2f s", where, counts, model_s)
                self._set_status(ICON_OK, "No issues found", reset=True)
                return
            text_access.apply(target, result.text)
            log.info("%s | %s | model %.2f s, total %.2f s", where, counts, model_s,
                     time.perf_counter() - started)
            self._set_status(ICON_OK, f"Fixed {result.changed_count} fragment(s)", reset=True)
        except (TextAccessError, OllamaError) as e:
            log.warning("%s | %s", where, e)
            self._ui(NSBeep)
            self._set_status(ICON_WARN, str(e), reset=True)
        except Exception as e:
            log.exception("%s | unexpected error", where)
            self._ui(NSBeep)
            self._set_status(ICON_WARN, f"Error: {e}", reset=True)
        finally:
            self._busy.release()


def main() -> None:
    debug = os.environ.get("GRAMMAR_FIX_DEBUG") == "1"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config()
    log.info("Grammar Fix starting: model %s, Ollama %s%s", cfg.model, cfg.ollama_url,
             " (DEBUG: text is logged)" if debug else "")
    # Accessory: no Dock icon and never steals focus from the app being edited.
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    GrammarFixApp(cfg).run()


if __name__ == "__main__":
    main()
