"""Mock local/remote runners: exercise the whole pipeline with NO live models.

This is a validation instrument, not a throwaway stub. Its knobs manufacture
easy/borderline/hard items and gates of known quality, so the harness's own
logic can be tested: a well-calibrated gate MUST reach the accuracy floor with
fewer remote tokens than a random one -- test_end_to_end asserts exactly that.

Answers (not scores) are emitted here and scored by the real scorer code, so a
mock run exercises scorers + records + frontier together.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List

from . import scorers
from .schema import Item, Record

DEFAULT_LOCAL = {"easy": 0.95, "borderline": 0.55, "hard": 0.20, "unknown": 0.60}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _wrong(item: Item, rng: random.Random):
    if item.task_type == "math":
        return str(int(item.gold) + rng.choice([-3, -2, -1, 1, 2, 5, 10]))
    if item.task_type == "classification":
        alts = [a for a in (item.allowed or ["neutral"]) if a != item.gold] or ["neutral"]
        return rng.choice(alts)
    toks = str(item.gold).split()                       # qa: partial or wrong span
    if toks and rng.random() < 0.5:
        return toks[0]
    return rng.choice(["Room A", "someone", "later today", "the manager"])


def build_records(items: List[Item],
                  local_competence: Dict[str, float] = None,
                  calib_noise: float = 0.6,
                  remote_competence: float = 0.92,
                  remote_hurts_rate: float = 0.06,
                  seed: int = 0) -> List[Record]:
    """Run mock local + remote on EVERY item (eval mode) and score both.

    remote_hurts_rate injects items where remote is wrong but local is right --
    these make the accuracy-vs-tokens curve non-monotone, which is real and
    which the frontier tracer must handle. Three confidence signals are emitted:
      informative -- correlated with local correctness (a good gate)
      random      -- pure noise (a useless gate)
      anti        -- correlated with INcorrectness (a broken gate)
    """
    comp = local_competence or DEFAULT_LOCAL
    rng = random.Random(seed + 7)
    out: List[Record] = []
    for it in items:
        # ---- local attempt ----
        p = comp.get(it.difficulty, 0.60)
        local_ok = rng.random() < p
        local_ans = it.gold if local_ok else _wrong(it, rng)
        local_score = scorers.score(local_ans, it)

        base = 1.0 if local_score >= 0.5 else -1.0     # higher conf => keep local
        informative = _sigmoid(2.0 * base + rng.gauss(0.0, calib_noise))
        randomsig = rng.random()
        anti = _sigmoid(-2.0 * base + rng.gauss(0.0, calib_noise))

        # ---- remote attempt ----
        if local_score >= 0.5 and rng.random() < remote_hurts_rate:
            remote_ok = False                          # remote worse than local
        else:
            remote_ok = rng.random() < remote_competence
        remote_ans = it.gold if remote_ok else _wrong(it, rng)
        remote_score = scorers.score(remote_ans, it)

        ptoks = len(it.input) // 4 + 8
        ctoks = {"math": 6, "classification": 2,
                 "qa": len(str(it.gold).split()) + 2}.get(it.task_type, 8)
        out.append(Record(
            id=it.id, task_type=it.task_type, difficulty=it.difficulty, gold=it.gold,
            local_answer=local_ans, local_score=local_score, local_tokens=ptoks + ctoks,
            remote_answer=remote_ans, remote_score=remote_score,
            remote_prompt_tokens=ptoks, remote_completion_tokens=ctoks,
            remote_total_tokens=ptoks + ctoks,
            confidences={"informative": informative, "random": randomsig, "anti": anti},
            input=it.input,
        ))
    return out
