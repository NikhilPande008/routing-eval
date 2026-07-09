"""The routing agent (production policy) and the real record-builder (calibration).

Router is what runs at scoring time: attempt local, ask the gate, escalate only
if confidence < tau. It shares runners and gates with the record-builder, so the
tau you calibrate on the frontier is the tau the agent uses.

build_records is the real counterpart to routing_eval.mock.build_records: it runs
local + remote on every item (eval mode) and emits P1 Records, so the exact same
frontier tracer sets tau on real data. Running remote on every item spends tokens
on the dev set ONCE (dev tokens are free toward the score).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import scorers
from .gates.signals import Gate, compute_confidences
from .llm.runners import LocalRunner, RemoteRunner
from .schema import Item, Record


@dataclass
class RouteResult:
    answer: str
    escalated: bool
    remote_tokens: int          # counts toward the score; 0 when kept local
    local_tokens: int
    confidence: float


class Router:
    def __init__(self, local: LocalRunner, remote: RemoteRunner, gate: Gate, tau: float):
        self.local, self.remote, self.gate, self.tau = local, remote, gate, tau

    def route(self, item: Item) -> RouteResult:
        lo = self.local.run(item)
        conf = self.gate.confidence(lo, item)
        if conf < self.tau:                       # escalate
            ro = self.remote.run(item)
            return RouteResult(ro.answer, True, ro.total_tokens, lo.tokens, conf)
        return RouteResult(lo.answer, False, 0, lo.tokens, conf)


def build_records(items: List[Item], local: LocalRunner, remote: RemoteRunner,
                  gates: List[Gate]) -> List[Record]:
    """Eval mode: run local + remote on EVERY item and score both, so the frontier
    can simulate any threshold. Produces the same Record schema the mock does."""
    out: List[Record] = []
    for it in items:
        lo = local.run(it)
        confs = compute_confidences(gates, lo, it)
        local_score = scorers.score(lo.answer, it)
        ro = remote.run(it)
        remote_score = scorers.score(ro.answer, it)
        out.append(Record(
            id=it.id, task_type=it.task_type, difficulty=it.difficulty, gold=it.gold,
            local_answer=lo.answer, local_score=local_score, local_tokens=lo.tokens,
            remote_answer=ro.answer, remote_score=remote_score,
            remote_prompt_tokens=ro.prompt_tokens,
            remote_completion_tokens=ro.completion_tokens,
            remote_total_tokens=ro.total_tokens,
            confidences=confs, input=it.input))
    return out
