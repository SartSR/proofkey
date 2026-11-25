"""Minimal Ollama client on the standard library (no extra dependency needed).

- temperature 0: correction must be deterministic, not creative.
- think=False: reasoning models (qwen3 family etc.) otherwise spend seconds on a
  hidden monologue. If the server/model rejects the field, we retry without it once
  and remember that.
- keep_alive: keeps the model in memory between hotkey presses (cold load = seconds).
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request

from .core import ChunkRejected
from .prompts import build_messages

log = logging.getLogger("grammar_fix")


class OllamaError(RuntimeError):
    pass


class _HTTPError(OllamaError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Ollama HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "",
        timeout: float = 60.0,
        keep_alive: str = "30m",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._no_think_field: set[str] = set()  # models that rejected the "think" field
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ http
    def _request(self, method: str, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise _HTTPError(e.code, e.read().decode("utf-8", "replace")) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise OllamaError(f"Ollama is not reachable at {self.base_url}: {e}") from None

    # ------------------------------------------------------------------- api
    def list_models(self) -> list[str]:
        data = self._request("GET", "/api/tags", timeout=5)
        return sorted(m["name"] for m in data.get("models", []))

    def warm_up(self) -> None:
        """Load the model into memory (an empty chat request only loads it)."""
        self._request("POST", "/api/chat", {"model": self.model, "messages": [], "keep_alive": self.keep_alive})

    def correct(self, text: str) -> str:
        model = self.model  # read once: the menu may switch models concurrently
        payload = {
            "model": model,
            "messages": build_messages(text),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                # English is ~4 chars/token; output ~= input. Generous cap, but a
                # runaway generation cannot block the hotkey for long.
                "num_predict": len(text) // 2 + 64,
            },
        }
        if model not in self._no_think_field:
            payload["think"] = False
        try:
            data = self._request("POST", "/api/chat", payload)
        except _HTTPError as e:
            if e.status == 400 and "think" in e.body.lower() and "think" in payload:
                with self._lock:
                    self._no_think_field.add(model)
                payload.pop("think")
                data = self._request("POST", "/api/chat", payload)
            else:
                raise
        if data.get("done_reason") == "length":
            # Hit num_predict: the answer is cut off mid-sentence. Accepting it
            # would silently delete the end of the user's text.
            log.warning("Model %s hit the token limit; chunk left unchanged", model)
            raise ChunkRejected("truncated")
        return (data.get("message") or {}).get("content", "")
