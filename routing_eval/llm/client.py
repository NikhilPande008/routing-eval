"""OpenAI-compatible chat client. Pure stdlib (urllib) -- keeps the package's
zero-runtime-dependency property. One real client (vLLM local, Fireworks remote,
same wire format) and one stub double for offline testing.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Talks to any /v1/chat/completions endpoint (vLLM, Fireworks, ...)."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, *, model: str, messages: List[Dict[str, str]], max_tokens: int,
             temperature: float = 0.0, stop: Optional[List[str]] = None, n: int = 1,
             logprobs: bool = False, top_logprobs: int = 0,
             extra: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages,
                                "max_tokens": max_tokens, "temperature": temperature, "n": n}
        if stop:
            body["stop"] = stop
        if logprobs:
            body["logprobs"] = True
            if top_logprobs:
                body["top_logprobs"] = top_logprobs
        if extra:
            body.update(extra)

        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"connection failed: {e.reason}") from e


class StubClient:
    """Test double. `handler` is either a fixed response dict or a callable that
    maps the request kwargs to a response dict. Records calls for assertions."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: List[Dict[str, Any]] = []

    def chat(self, **kwargs) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return self.handler(kwargs) if callable(self.handler) else self.handler


def stub_response(contents: List[str],
                  token_logprobs: Optional[List[List[float]]] = None,
                  usage: Optional[Dict[str, int]] = None,
                  finish_reason: str = "stop") -> Dict[str, Any]:
    """Build an OpenAI-shaped response. `contents` is one string per choice;
    `token_logprobs[i]` is the per-token logprob list for choice i.
    `finish_reason` defaults to "stop"; pass "length" to simulate a truncated
    answer (see policy.py's retry-on-truncation logic, D40)."""
    choices = []
    for i, content in enumerate(contents):
        choice: Dict[str, Any] = {"index": i, "finish_reason": finish_reason,
                                  "message": {"role": "assistant", "content": content}}
        if token_logprobs is not None:
            choice["logprobs"] = {"content": [{"token": f"t{j}", "logprob": lp}
                                              for j, lp in enumerate(token_logprobs[i])]}
        choices.append(choice)
    resp: Dict[str, Any] = {"choices": choices}
    if usage:
        resp["usage"] = usage
    return resp
