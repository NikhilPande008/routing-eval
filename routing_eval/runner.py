"""Orchestration: dataset -> Records. Defines the P1/P2 boundary.

The ModelRunner protocol is what P2 implements with a real vLLM local endpoint
and a real Fireworks remote endpoint; routing_eval.mock implements it for
testing. In eval mode we run remote on EVERY item so the frontier can simulate
any threshold -- this spends real remote tokens on your whole dev set ONCE
(dev tokens are free toward the score). After that, sweeping the threshold and
comparing gates costs zero tokens (record-then-replay). Cache records.json and
never re-pay.
"""
from __future__ import annotations

from typing import Protocol

from .schema import Item


class LocalResult(Protocol):
    answer: object
    tokens: int
    confidences: dict


class RemoteResult(Protocol):
    answer: object
    prompt_tokens: int
    completion_tokens: int


class ModelRunner(Protocol):
    def run_local(self, item: Item) -> LocalResult: ...
    def run_remote(self, item: Item) -> RemoteResult: ...


# Real runners are P2. They MUST fill the same Record schema. Reference sketch
# of the token-minimal remote call (see README for the full rationale):
#
#   resp = client.chat.completions.create(
#       model="accounts/fireworks/models/<revealed>",
#       messages=messages,
#       max_tokens=ANSWER_CAP,          # tightest the answer format allows
#       stop=STOP_SEQUENCES,            # halt the instant the answer is done
#       temperature=0,
#       extra_body={"reasoning_effort": "none"},   # reasoning tokens are billed
#       logprobs=1,                     # feeds the logprob gate signal
#   )
#   usage = resp.usage   # -> remote_prompt_tokens / completion_tokens / total_tokens
#
# Until P2 lands, use routing_eval.mock.build_records for a full run.
