"""Read the focused text field of any app and replace its text.

Strategy (chosen for reach and safety, not elegance):
- READ via the Accessibility API (AXUIElement): selected text if any, otherwise
  the whole field value. Non-destructive and tells us if it is a password field.
  Electron apps (Slack, Teams, VS Code...) only expose their AX tree after the
  AXManualAccessibility attribute is set, so we set it when the focus is missing.
- If AX cannot reach a focused element at all, fall back to the clipboard:
  Cmd+C (or Cmd+A, Cmd+C) and read the pasteboard.
- WRITE always by pasting (Cmd+V). Setting AXValue directly bypasses web apps'
  input events (React editors in Teams/Slack ignore it) and breaks Undo; a paste
  is handled by every app like user input and can be undone with Cmd+Z.
- The user's clipboard is saved and restored. Our temporary content is marked
  transient/concealed so clipboard managers ignore it.
- Before pasting we re-check the app and the text: if the user switched apps or
  kept typing while the model was working, nothing is replaced.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import ApplicationServices as AS
import Quartz
from AppKit import NSPasteboard, NSPasteboardItem, NSPasteboardTypeString, NSWorkspace
from Foundation import NSData

KEY_A, KEY_C, KEY_V = 0, 8, 9
OBJECT_REPLACEMENT_CHAR = "￼"  # embedded mention/image/attachment in rich text
_TRANSIENT_TYPES = ("org.nspasteboard.TransientType", "org.nspasteboard.ConcealedType")
_MODIFIER_MASK = (
    Quartz.kCGEventFlagMaskCommand
    | Quartz.kCGEventFlagMaskAlternate
    | Quartz.kCGEventFlagMaskControl
    | Quartz.kCGEventFlagMaskShift
)
_SECURE_ROLES = {"AXSecureTextField"}


class TextAccessError(Exception):
    """Expected, user-facing failure (shown in the menu bar, nothing was changed)."""


@dataclass
class Target:
    pid: int
    element: Any | None  # AXUIElement, or None when only the clipboard worked
    text: str
    whole_field: bool  # True: replace the whole field; False: only the selection
    app_name: str = ""

    @property
    def via(self) -> str:
        return "AX" if self.element is not None else "clipboard"

    @property
    def mode(self) -> str:
        return "whole field" if self.whole_field else "selection"


# ------------------------------------------------------------------ permissions


def is_trusted(prompt: bool = False) -> bool:
    """Accessibility permission (needed for AX reads and synthetic key presses)."""
    return bool(AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: prompt}))


# ------------------------------------------------------------------------- AX


def _ax_get(element: Any, attribute: str) -> Any:
    err, value = AS.AXUIElementCopyAttributeValue(element, attribute, None)
    return value if err == AS.kAXErrorSuccess else None


def _frontmost_pid() -> int:
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return int(app.processIdentifier()) if app else -1


def _frontmost_name() -> str:
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return str(app.localizedName() or "") if app else ""


def _focused_element(pid: int) -> Any | None:
    app = AS.AXUIElementCreateApplication(pid)
    element = _ax_get(app, AS.kAXFocusedUIElementAttribute)
    if element is None:
        # Electron/Chromium apps build their AX tree only on request.
        AS.AXUIElementSetAttributeValue(app, "AXManualAccessibility", True)
        time.sleep(0.1)
        element = _ax_get(app, AS.kAXFocusedUIElementAttribute)
    return element


def _is_secure(element: Any) -> bool:
    role = _ax_get(element, AS.kAXRoleAttribute)
    subrole = _ax_get(element, AS.kAXSubroleAttribute)
    return role in _SECURE_ROLES or subrole in _SECURE_ROLES


# ------------------------------------------------------------------- keyboard


def _wait_modifiers_released(timeout: float = 2.0) -> None:
    """The user may still hold the hotkey modifiers; they would corrupt Cmd+V."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateHIDSystemState) & _MODIFIER_MASK:
            return
        time.sleep(0.02)


def _press_cmd(keycode: int) -> None:
    source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(source, keycode, down)
        Quartz.CGEventSetFlags(event, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.01)


# ------------------------------------------------------------------ clipboard


class _Clipboard:
    def __init__(self) -> None:
        self.pb = NSPasteboard.generalPasteboard()
        self._saved: list[list[tuple[str, Any]]] = []
        self._our_change_count: int | None = None

    def save(self) -> None:
        self._saved = []
        for item in self.pb.pasteboardItems() or []:
            entry = []
            for type_ in item.types():
                data = item.dataForType_(type_)
                if data is not None:
                    entry.append((type_, data))
            self._saved.append(entry)

    def restore(self) -> None:
        # Do not clobber something the user copied while we were working.
        if self._our_change_count is not None and self.pb.changeCount() != self._our_change_count:
            return
        self.pb.clearContents()
        items = []
        for entry in self._saved:
            item = NSPasteboardItem.alloc().init()
            for type_, data in entry:
                item.setData_forType_(data, type_)
            items.append(item)
        if items:
            self.pb.writeObjects_(items)

    def put_text(self, text: str) -> None:
        self.pb.declareTypes_owner_([NSPasteboardTypeString, *_TRANSIENT_TYPES], None)
        self.pb.setString_forType_(text, NSPasteboardTypeString)
        for type_ in _TRANSIENT_TYPES:
            self.pb.setData_forType_(NSData.data(), type_)
        self._our_change_count = self.pb.changeCount()

    def copy_selection(self, timeout: float = 0.5) -> str | None:
        before = self.pb.changeCount()
        _press_cmd(KEY_C)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.pb.changeCount() != before:
                self._our_change_count = self.pb.changeCount()
                return self.pb.stringForType_(NSPasteboardTypeString)
            time.sleep(0.02)
        return None  # nothing was selected


# ---------------------------------------------------------------------- public


def capture() -> Target:
    """Grab the text to correct from the focused field of the frontmost app."""
    _wait_modifiers_released()
    pid = _frontmost_pid()
    name = _frontmost_name()
    if pid <= 0 or pid == os.getpid():
        raise TextAccessError("No target app in focus")

    element = _focused_element(pid)
    if element is not None:
        if _is_secure(element):
            raise TextAccessError(f"{name}: password field, skipped")
        selected = _ax_get(element, AS.kAXSelectedTextAttribute)
        if isinstance(selected, str) and selected.strip():
            return Target(pid, element, str(selected), whole_field=False, app_name=name)
        value = _ax_get(element, AS.kAXValueAttribute)
        if isinstance(value, str):
            if not value.strip():
                raise TextAccessError(f"{name}: field is empty")
            if OBJECT_REPLACEMENT_CHAR in value:
                # Re-pasting plain text would destroy mentions/images in rich editors.
                raise TextAccessError(f"{name}: field has mentions/attachments, select the text to fix")
            return Target(pid, element, str(value), whole_field=True, app_name=name)
        # Focus is on something that is not a text field (a button, a list...):
        # Cmd+A there could select files or a whole page, so do not fall back.
        role = _ax_get(element, AS.kAXRoleAttribute)
        raise TextAccessError(f"{name}: focused element ({role}) is not a text field")

    return _capture_via_clipboard(pid, name)


def _capture_via_clipboard(pid: int, name: str) -> Target:
    clipboard = _Clipboard()
    clipboard.save()
    try:
        text = clipboard.copy_selection()
        whole = False
        if not text:
            _press_cmd(KEY_A)
            time.sleep(0.05)
            text = clipboard.copy_selection()
            whole = True
    finally:
        clipboard.restore()
    if not text or not text.strip():
        raise TextAccessError(f"{name}: no text found in the focused field")
    return Target(pid, None, text, whole_field=whole, app_name=name)


def _current_text(target: Target) -> Any:
    attr = AS.kAXValueAttribute if target.whole_field else AS.kAXSelectedTextAttribute
    return _ax_get(target.element, attr)


def apply(target: Target, new_text: str) -> None:
    """Replace the captured text with `new_text` by pasting it."""
    _wait_modifiers_released()
    if _frontmost_pid() != target.pid:
        raise TextAccessError("You switched apps: nothing replaced")
    if target.element is not None and _current_text(target) != target.text:
        raise TextAccessError("Text changed while checking: nothing replaced")

    clipboard = _Clipboard()
    clipboard.save()
    try:
        if target.whole_field:
            _press_cmd(KEY_A)
            time.sleep(0.05)
        clipboard.put_text(new_text)
        _press_cmd(KEY_V)
        _wait_paste_consumed(target)
    finally:
        clipboard.restore()


def _wait_paste_consumed(target: Target, timeout: float = 2.0) -> None:
    """Restoring the clipboard too early would paste the OLD clipboard content.

    With AX we wait until the field actually changed; without AX a fixed delay
    is the best we can do.
    """
    if target.element is None:
        time.sleep(0.5)
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _current_text(target) != target.text:
            time.sleep(0.05)
            return
        time.sleep(0.03)
