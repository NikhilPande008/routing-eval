"""Synthetic, correct-by-construction stand-in tasks.

Purpose: battle-test the harness BEFORE the real task exists. Difficulty tiers
create a dense 'borderline' band -- the region where the escalate/keep decision
actually breaks (last hackathon's lesson: happy-path testing hides the router
bug). Golds are computed or extracted, never asserted, so nothing here can
encode a wrong fact.
"""
from __future__ import annotations

import random
from typing import List

from .schema import Item

_POS = ["excellent", "fantastic", "wonderful", "great", "superb", "delightful"]
_NEG = ["terrible", "awful", "broken", "useless", "disappointing", "clumsy"]
_NAMES = ["Alice", "Ben", "Carol", "Dan", "Eve", "Frank", "Grace", "Hana"]
_ROOMS = ["Room A", "Room B", "Room C", "the Atrium", "the Lab", "the Annex"]


def _difficulty(k: int, _n: int) -> str:
    # dense borderline band: ~30% easy, ~45% borderline, ~25% hard
    r = (k % 20) / 20.0
    if r < 0.30:
        return "easy"
    if r < 0.75:
        return "borderline"
    return "hard"


def make_math(n: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed)
    out = []
    for k in range(n):
        d = _difficulty(k, n)
        if d == "easy":
            a, b = rng.randint(2, 20), rng.randint(2, 20)
            gold, q = a + b, f"What is {a} plus {b}?"
        elif d == "borderline":
            a, b, c = (rng.randint(5, 40) for _ in range(3))
            gold = a * b - c
            q = f"A box holds {a} rows of {b} items. If {c} items are removed, how many remain?"
        else:
            a, b, c, e = (rng.randint(6, 30) for _ in range(4))
            gold = a * b + c * e
            q = (f"One crate has {a} shelves of {b} parts; another has {c} shelves "
                 f"of {e} parts. How many parts in total?")
        out.append(Item(f"math-{k}", "math", q, gold, "numeric", {"rel_tol": 1e-6}, d))
    return out


def make_classification(n: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed + 1)
    allowed = ["positive", "negative", "neutral"]
    out = []
    for k in range(n):
        d = _difficulty(k, n)
        if d == "easy":
            if rng.random() < 0.5:
                w = rng.choice(_POS)
                gold, s = "positive", f"An {w} experience, {w} in every single way."
            else:
                w = rng.choice(_NEG)
                gold, s = "negative", f"A {w} experience, {w} from start to finish."
        elif d == "borderline":
            p, ng = rng.choice(_POS), rng.choice(_NEG)
            gold, s = "positive", f"The product itself is {p}, though the manual was a bit {ng}."
        else:
            ng = rng.choice(_NEG)
            gold, s = "positive", f"I fully expected it to be {ng}, but it was not {ng} at all."
        out.append(Item(f"cls-{k}", "classification", s, gold,
                        "multiple_choice", {}, d, allowed=allowed))
    return out


def make_qa(n: int, seed: int = 0) -> List[Item]:
    rng = random.Random(seed + 2)
    out = []
    for k in range(n):
        d = _difficulty(k, n)
        who = rng.sample(_NAMES, 3)
        where = rng.sample(_ROOMS, 3)
        hour = rng.randint(1, 12)
        if d == "easy":
            ctx = f"{who[0]} has a meeting at {hour} in {where[0]}."
            q, gold = "Who has the meeting?", who[0]
        elif d == "borderline":
            ctx = (f"{who[0]} is in {where[0]}. {who[1]} is in {where[1]}. "
                   f"The {hour} o'clock review is led by {who[2]} in {where[2]}.")
            q, gold = "Who leads the review?", who[2]
        else:
            ctx = (f"{who[0]} and {who[1]} met earlier in {where[0]}. "
                   f"Later {who[2]} moved to {where[1]} while {who[0]} stayed. "
                   f"The report is due to {who[1]}, not {who[2]}.")
            q, gold = "Who is the report due to?", who[1]
        out.append(Item(f"qa-{k}", "qa", f"Context: {ctx} Question: {q}",
                        gold, "token_f1", {}, d))
    return out


def make_standin(n_per: int = 100, seed: int = 0) -> List[Item]:
    """Mixed math + classification + qa, with a dense borderline band."""
    return (make_math(n_per, seed)
            + make_classification(n_per, seed)
            + make_qa(n_per, seed))
