"""Data contract between the harness (P1) and the router (P2).

A Record is the atomic unit the frontier tracer consumes. P2's job is to
produce these Records from real models; this harness only consumes them.
Everything the tracer needs to simulate ANY escalation threshold offline lives
in one Record, so the expensive step (running local+remote once) is cleanly
separated from the cheap step (sweeping the threshold).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Item:
    """One evaluation task with a gold answer and how to score it."""
    id: str
    task_type: str                        # "math" | "classification" | "qa" | ...
    input: str
    gold: Any
    scorer: str                           # key into scorers.SCORERS
    scorer_opts: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "unknown"           # easy | borderline | hard (analysis only)
    allowed: Optional[List[str]] = None   # label set for classification


@dataclass
class Record:
    """Outcome of running local + remote on one Item, plus gate signals.

    local_score / remote_score are floats in [0,1] (1.0 = correct, or graded).
    remote_total_tokens is what the competition counts if this item escalates.
    confidences maps each candidate gate signal -> a score where HIGHER means
    'keep local' (less likely to escalate). A native signal pointing the other
    way (entropy, perplexity) must be negated before it is stored here.
    """
    id: str
    task_type: str
    difficulty: str
    gold: Any
    local_answer: Any
    local_score: float
    local_tokens: int                     # 0 toward the score; kept as a latency proxy
    remote_answer: Any
    remote_score: float
    remote_prompt_tokens: int
    remote_completion_tokens: int
    remote_total_tokens: int
    confidences: Dict[str, float]
    input: Optional[str] = None


def save_records(records: List[Record], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)


def load_records(path: str) -> List[Record]:
    with open(path) as f:
        raw = json.load(f)
    return [Record(**r) for r in raw]
