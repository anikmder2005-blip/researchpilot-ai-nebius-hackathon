"""OpenAI-compatible chat client for Nebius Token Factory (default) or NVIDIA NIM.

Uses plain `requests` against `/chat/completions` so there is one small, auditable code path.
The provider is always the one chosen in configuration; there is no silent fallback.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

import requests

from config.settings import Settings, redact

log = logging.getLogger("researchpilot.llm")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    """Raised for any model-call failure. Messages never contain secrets."""


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks that some reasoning models emit inline."""
    text = _THINK_RE.sub("", text or "")
    # Unterminated think block (budget exhausted mid-reasoning): drop the dangling part.
    if "<think>" in text.lower():
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


def extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from model text. Returns None if absent/invalid."""
    text = strip_reasoning(text)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


class LLMClient:
    def __init__(self, settings: Settings, session: Optional[requests.Session] = None, max_retries: int = 2):
        self.settings = settings
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.calls = 0  # real number of completed requests (shown in UI/metrics)

    @property
    def model(self) -> str:
        return self.settings.llm_model

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> str:
        s = self.settings
        if not s.llm_ready:
            raise LLMError(f"{s.key_env_name} is not configured. See .env.example.")
        url = s.llm_base_url + "chat/completions"
        body = {
            "model": s.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or s.max_output_tokens,
        }
        headers = {"Authorization": f"Bearer {s.llm_api_key}", "Content-Type": "application/json"}

        last_err = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(url, headers=headers, json=body, timeout=s.request_timeout_s)
            except requests.Timeout:
                last_err = f"request timed out after {s.request_timeout_s}s"
            except requests.RequestException as exc:
                last_err = f"network error: {redact(str(exc), s.llm_api_key)}"
            else:
                if resp.status_code == 200:
                    self.calls += 1
                    return self._parse(resp)
                last_err = self._http_error(resp)
                # Only retry rate limits and transient server errors.
                if resp.status_code not in (429, 500, 502, 503, 504):
                    break
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 4))
        log.warning("LLM call failed: %s", last_err)
        raise LLMError(f"Model request to {s.provider_label} failed: {last_err}")

    def _http_error(self, resp: Any) -> str:
        code = resp.status_code
        hint = {
            401: "authentication failed (check your API key)",
            403: "access denied for this key or model",
            404: "model or endpoint not found (check the model ID in .env)",
            429: "rate limited",
        }.get(code, "server error" if code >= 500 else "request rejected")
        detail = ""
        try:
            err = resp.json().get("error", {})
            detail = err.get("message", "") if isinstance(err, dict) else str(err)
        except Exception:
            detail = ""
        detail = redact(detail, self.settings.llm_api_key)[:200]
        return f"HTTP {code} - {hint}" + (f": {detail}" if detail else "")

    def _parse(self, resp: Any) -> str:
        try:
            data = resp.json()
            choice = data["choices"][0]
            content = (choice.get("message") or {}).get("content")
            finish = choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("Model returned a malformed response (no choices/message).")
        text = strip_reasoning(content or "")
        if not text:
            if finish == "length":
                raise LLMError(
                    "The model ran out of output tokens before producing an answer "
                    "(reasoning models spend tokens thinking). Increase MAX_OUTPUT_TOKENS."
                )
            raise LLMError("Model returned an empty answer.")
        return text

    def chat_json(self, messages: list[dict[str, str]], max_tokens: Optional[int] = None) -> Optional[dict]:
        """Chat and parse a JSON object. Returns None when the model output is not valid JSON."""
        return extract_json(self.chat(messages, max_tokens=max_tokens, temperature=0.1))
