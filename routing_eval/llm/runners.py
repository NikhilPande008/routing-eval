"""Local (vLLM) and remote (Fireworks) runners. Each turns an Item into a
structured output the gates and the record-builder consume.

2026-07-10 (D40): `max_tokens` defaults raised 32 -> 512. The 32-token default
was a token-minimization leftover from before the accuracy gate existed; a
32-token cap silently truncates any multi-sentence explanation, which is a
guaranteed judge failure (D18: the gate is an LLM judge grading intent, not a
token-count optimizer) -- token headroom is not the binding constraint right
now (the leaderboard leader spends ~225 tokens/task; D33's baseline spends
~95), truncation is. `RemoteRunner` also now records `finish_reason` on every
call so a truncated (`finish_reason="length"`) answer can be detected and
retried with a doubled cap -- see `policy.py`'s `_try_fireworks`.
temperature=0, reasoning_effort='none' (reasoning tokens are billed) are kept
-- those don't trade off against answer completeness the way max_tokens does.
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
    finish_reason: Optional[str] = None   # "length" means the answer was truncated
    raw: Dict[str, Any] = field(default_factory=dict)


class LocalRunner:
    """vLLM local model. For gate comparison, set n_samples>1 and temperature>0
    so the self-consistency gate has variation; the logprob gate reads choice 0.
    Local tokens are free here, so running local more than once is fine."""

    def __init__(self, client, model: str, max_tokens: int = 512, temperature: float = 0.0,
                 stop: Optional[List[str]] = None, n_samples: int = 1,
                 system: Optional[str] = None, logprobs: bool = True):
        self.client, self.model = client, model
        self.max_tokens, self.temperature = max_tokens, temperature
        self.stop, self.n_samples, self.system = stop, n_samples, system
        self.logprobs = logprobs   # gates need them; the deployed score path doesn't

    def run(self, item: Item, timeout: Optional[float] = None,
            system: Optional[str] = None,
            temperature: Optional[float] = None) -> LocalOutput:
        """`timeout` (seconds) forwards to the client's own socket timeout --
        real protection against a wedged local server. routing_eval.policy
        additionally bounds this call in a thread for clients (like tests'
        StubClient) that don't do real I/O and so can't honor it themselves.

        `system` (2026-07-11, local tier) overrides the constructor-level
        system prompt per call -- the policy router passes each entry's
        template here so one LocalRunner serves every local category. None
        keeps the constructor value (which itself defaults to the runner-level
        DEFAULT_SYSTEM via build_messages)."""
        resp = self.client.chat(
            model=self.model, messages=build_messages(item, system if system is not None
                                                      else self.system),
            max_tokens=self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            stop=self.stop,
            n=self.n_samples, logprobs=self.logprobs, timeout=timeout)
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

    def __init__(self, client, model: str, max_tokens: int = 512,
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
            token_logprobs=_token_logprobs(choice),
            finish_reason=choice.get("finish_reason"), raw=resp)
