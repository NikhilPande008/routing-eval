"""Local (vLLM) and remote (Fireworks) runners. Each turns an Item into a
structured output the gates and the record-builder consume.

Token-minimization levers on the remote call (billed): tight max_tokens, stop
sequences, temperature 0, reasoning_effort='none' (reasoning tokens are billed).
The exact answer cap and stop sequences are tuned to the revealed scorer's answer
format at kickoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schema import Item

DEFAULT_SYSTEM = "Answer with only the final answer, as briefly as possible. Do not explain."


def build_messages(item: Item, system: Optional[str] = None) -> List[Dict[str, str]]:
    msgs = []
    s = DEFAULT_SYSTEM if system is None else system
    if s:
        msgs.append({"role": "system", "content": s})
    msgs.append({"role": "user", "content": item.input})
    return msgs


def _content(choice: Dict[str, Any]) -> str:
    return choice["message"]["content"]


def _token_logprobs(choice: Dict[str, Any]) -> List[float]:
    lp = choice.get("logprobs") or {}
    return [tok["logprob"] for tok in (lp.get("content") or [])]


@dataclass
class LocalOutput:
    answer: str
    token_logprobs: List[float]           # per-token logprob of choice 0
    samples: List[str]                    # all n sampled answers (self-consistency)
    tokens: int                           # completion tokens (free; logged)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RemoteOutput:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    token_logprobs: List[float]
    raw: Dict[str, Any] = field(default_factory=dict)


class LocalRunner:
    """vLLM local model. For gate comparison, set n_samples>1 and temperature>0
    so the self-consistency gate has variation; the logprob gate reads choice 0.
    Local tokens are free here, so running local more than once is fine."""

    def __init__(self, client, model: str, max_tokens: int = 32, temperature: float = 0.0,
                 stop: Optional[List[str]] = None, n_samples: int = 1,
                 system: Optional[str] = None):
        self.client, self.model = client, model
        self.max_tokens, self.temperature = max_tokens, temperature
        self.stop, self.n_samples, self.system = stop, n_samples, system

    def run(self, item: Item, timeout: Optional[float] = None) -> LocalOutput:
        """`timeout` (seconds) forwards to the client's own socket timeout --
        real protection against a wedged local server. routing_eval.policy
        additionally bounds this call in a thread for clients (like tests'
        StubClient) that don't do real I/O and so can't honor it themselves."""
        resp = self.client.chat(
            model=self.model, messages=build_messages(item, self.system),
            max_tokens=self.max_tokens, temperature=self.temperature, stop=self.stop,
            n=self.n_samples, logprobs=True, timeout=timeout)
        choices = resp["choices"]
        usage = resp.get("usage") or {}
        return LocalOutput(
            answer=_content(choices[0]),
            token_logprobs=_token_logprobs(choices[0]),
            samples=[_content(ch) for ch in choices],
            tokens=usage.get("completion_tokens", 0),
            raw=resp)


class RemoteRunner:
    """Fireworks remote model, tuned for minimum billed tokens."""

    def __init__(self, client, model: str, max_tokens: int = 32,
                 stop: Optional[List[str]] = None, reasoning_effort: Optional[str] = "none",
                 system: Optional[str] = None, logprobs: bool = False):
        self.client, self.model = client, model
        self.max_tokens, self.stop = max_tokens, stop
        self.reasoning_effort, self.system, self.logprobs = reasoning_effort, system, logprobs

    def run(self, item: Item, timeout: Optional[float] = None) -> RemoteOutput:
        """`timeout` (seconds) forwards to the client's own socket timeout --
        real protection against a wedged/slow Fireworks call, same pattern as
        LocalRunner.run(). Matters more now that routing_eval.policy retries
        a blank/failed answer with a second model: without a bound here, two
        uncapped calls could each take the client's full default timeout."""
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else None
        resp = self.client.chat(
            model=self.model, messages=build_messages(item, self.system),
            max_tokens=self.max_tokens, temperature=0.0, stop=self.stop, n=1,
            logprobs=self.logprobs, extra=extra, timeout=timeout)
        choice = resp["choices"][0]
        usage = resp.get("usage") or {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        return RemoteOutput(
            answer=_content(choice), prompt_tokens=pt, completion_tokens=ct,
            total_tokens=usage.get("total_tokens", pt + ct),
            token_logprobs=_token_logprobs(choice), raw=resp)
