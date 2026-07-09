"""Gate signals: each maps a local model output to a confidence in the LOCAL
answer, where HIGHER means keep local (the convention the frontier tracer
expects). All three here are free -- computed from the local output, no extra
model call.

A gate menu, chosen empirically per task via the frontier:
  LogprobGate         cheapest; single-pass token confidence. Insufficient alone.
  SelfConsistencyGate strong; needs n>1 samples (free locally here). N x latency.
  DeterministicGate   free and precise where the answer has checkable structure.
Layer them (deterministic veto -> logprob/consistency) and let the frontier pick.

An LLM-as-judge gate is deliberately omitted: it needs a second model call, so it
is a billed feature, not a free local signal. Add it in router logic if the
frontier shows it earns its tokens.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import List, Protocol

from .. import scorers
from ..llm.runners import LocalOutput
from ..schema import Item


class Gate(Protocol):
    name: str
    def confidence(self, local: LocalOutput, item: Item) -> float: ...


class LogprobGate:
    """Geometric-mean token probability of the local answer, in [0,1]."""
    name = "logprob"

    def confidence(self, local: LocalOutput, item: Item) -> float:
        lps = local.token_logprobs
        if not lps:
            return 0.0
        return math.exp(sum(lps) / len(lps))


def _bucket(answer: str, item: Item) -> str:
    """Task-aware normalization so equivalent answers group together."""
    if item.task_type == "math" or item.scorer == "numeric":
        nums = scorers._NUM.findall(str(answer))
        return nums[-1].replace(",", "").replace("$", "") if nums else "<none>"
    if item.scorer == "multiple_choice" and item.allowed:
        low = answer.casefold()
        hits = [lab for lab in item.allowed if re.search(r"\b" + re.escape(lab.casefold()) + r"\b", low)]
        return hits[-1].casefold() if hits else "<none>"
    return scorers._norm(answer, strip_punct=True, drop_articles=True)


class SelfConsistencyGate:
    """Fraction of samples agreeing with the modal answer. Needs n>1 samples
    (run the local model with n_samples>1 and temperature>0)."""
    name = "self_consistency"

    def confidence(self, local: LocalOutput, item: Item) -> float:
        if not local.samples:
            return 0.0
        buckets = [_bucket(s, item) for s in local.samples]
        return Counter(buckets).most_common(1)[0][1] / len(buckets)


class DeterministicGate:
    """1.0 if the answer has valid structure for the task, else 0.0. Most useful
    as a veto layer for structured outputs (numeric, choice, json)."""
    name = "deterministic"

    def confidence(self, local: LocalOutput, item: Item) -> float:
        a = str(local.answer).strip()
        if item.scorer == "numeric":
            return 1.0 if scorers._NUM.search(a) else 0.0
        if item.scorer == "multiple_choice" and item.allowed:
            low = a.casefold()
            return 1.0 if any(re.search(r"\b" + re.escape(l.casefold()) + r"\b", low)
                              for l in item.allowed) else 0.0
        if item.scorer == "json_match":
            try:
                json.loads(re.sub(r"^```(?:json)?|```$", "", a, flags=re.MULTILINE).strip())
                return 1.0
            except json.JSONDecodeError:
                return 0.0
        return 1.0 if a else 0.0


FREE_GATES = [DeterministicGate(), LogprobGate(), SelfConsistencyGate()]
GATES = {g.name: g for g in FREE_GATES}


def compute_confidences(gates: List[Gate], local: LocalOutput, item: Item) -> dict:
    """Every gate's confidence from one local output -> the record's confidences map."""
    return {g.name: g.confidence(local, item) for g in gates}
