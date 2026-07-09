"""The frontier tracer: the instrument that sets the escalation threshold.

Given per-item Records (local score, remote score, remote token cost, and one
confidence signal), it sweeps every achievable threshold and reports the
accuracy-vs-remote-tokens curve, plus:
  - all-local / all-remote baselines
  - the union ceiling: the max accuracy ANY per-item router can reach
  - the oracle: min tokens to hit an accuracy floor with perfect knowledge
  - the operating point: min tokens to hit the floor with THIS gate
  - gate efficiency = oracle_tokens / gate_tokens (<=1; how good the gate is)

Because local tokens are free in this competition, the optimal policy sits at
the aggressive edge -- just above the accuracy floor. You cannot find that edge
without this curve, which is why the harness is priority one.

Convention: confidence is confidence in the LOCAL answer. Higher => keep local.
Escalate item i iff conf[i] < tau.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class FrontierPoint:
    tau: float
    remote_tokens: int
    accuracy: float
    escalation_rate: float


@dataclass
class FrontierResult:
    signal: str
    points: List[FrontierPoint]           # sorted by remote_tokens ascending
    all_local_accuracy: float
    all_remote_accuracy: float
    all_remote_tokens: int
    union_ceiling: float
    accuracy_floor: Optional[float] = None
    operating_point: Optional[FrontierPoint] = None
    feasible: bool = True
    oracle_tokens: Optional[int] = None
    oracle_exact: bool = True
    gate_efficiency: Optional[float] = None


def _cols(records):
    return ([r.local_score for r in records],
            [r.remote_score for r in records],
            [r.remote_total_tokens for r in records])


def trace_frontier(records, signal: str) -> FrontierResult:
    if not records:
        raise ValueError("no records to trace")
    conf = [r.confidences[signal] for r in records]
    local, remote, cost = _cols(records)
    n = len(records)

    taus = sorted(set(conf))
    taus.append(taus[-1] + 1.0)           # a threshold above max conf => escalate all
    pts: List[FrontierPoint] = []
    for tau in taus:
        tok = esc = 0
        tot = 0.0
        for i in range(n):
            if conf[i] < tau:             # escalate
                tot += remote[i]
                tok += cost[i]
                esc += 1
            else:                         # keep local
                tot += local[i]
        pts.append(FrontierPoint(tau, tok, tot / n, esc / n))
    pts.sort(key=lambda p: (p.remote_tokens, p.escalation_rate))

    return FrontierResult(
        signal=signal,
        points=pts,
        all_local_accuracy=sum(local) / n,
        all_remote_accuracy=sum(remote) / n,
        all_remote_tokens=sum(cost),
        union_ceiling=sum(max(local[i], remote[i]) for i in range(n)) / n,
    )


def oracle_tokens(records, floor: float) -> Tuple[Optional[int], bool]:
    """Min remote tokens to reach the accuracy floor with perfect knowledge.

    Exact when all scores are binary (escalate the cheapest local-wrong/
    remote-right items). For graded scores it is a greedy gain-per-token
    approximation (an achievable upper bound), flagged via the returned bool.
    Returns (None, _) if even an oracle cannot reach the floor -> you are
    capability-limited, not routing-limited.
    """
    local, remote, cost = _cols(records)
    n = len(records)
    binary = all(s in (0.0, 1.0) for s in local + remote)
    if sum(local) / n >= floor - 1e-12:
        return 0, binary
    if sum(max(local[i], remote[i]) for i in range(n)) / n < floor - 1e-12:
        return None, binary
    gains = [(remote[i] - local[i], cost[i]) for i in range(n) if remote[i] > local[i]]
    gains.sort(key=lambda gc: (gc[0] / gc[1]) if gc[1] > 0 else float("inf"), reverse=True)
    cur, target, tok = sum(local), floor * n, 0
    for g, c in gains:
        if cur >= target - 1e-9:
            break
        cur += g
        tok += c
    return tok, binary


def add_operating_point(fr: FrontierResult, records, floor: float) -> FrontierResult:
    fr.accuracy_floor = floor
    feasible = [p for p in fr.points if p.accuracy >= floor - 1e-12]
    if not feasible:
        fr.feasible = False
        fr.operating_point = None
    else:
        fr.feasible = True
        # min tokens; tie-break toward more accuracy margin, then fewer escalations
        fr.operating_point = min(
            feasible, key=lambda p: (p.remote_tokens, -p.accuracy, p.escalation_rate))
    ot, exact = oracle_tokens(records, floor)
    fr.oracle_tokens, fr.oracle_exact = ot, exact
    if fr.operating_point is not None and ot is not None:
        gt = fr.operating_point.remote_tokens
        fr.gate_efficiency = 1.0 if gt == 0 else ot / gt
    return fr


def evaluate(records, signal: str, floor: float) -> FrontierResult:
    """Trace the frontier and locate the operating point for an accuracy floor."""
    return add_operating_point(trace_frontier(records, signal), records, floor)
