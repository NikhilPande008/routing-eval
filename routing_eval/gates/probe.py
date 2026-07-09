"""Learned probe (the FrugalGPT-style correctness predictor) -- the most powerful
gate, but it must be FIT on labeled dev records, so it comes online only after
kickoff. It combines the free base signals into one calibrated confidence.

Pure stdlib: a small logistic regression trained by batch gradient descent. It
predicts P(local answer correct) from the base gates' confidences; at inference
that probability IS the "keep local" confidence.
"""
from __future__ import annotations

import math
from typing import List, Sequence

from ..llm.runners import LocalOutput
from ..schema import Item
from .signals import Gate, compute_confidences


class LogisticProbe:
    def __init__(self):
        self.w: List[float] = []
        self.b: float = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float],
            epochs: int = 400, lr: float = 0.3) -> "LogisticProbe":
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(epochs):
            gw, gb = [0.0] * d, 0.0
            for xi, yi in zip(X, y):
                p = self.prob(xi)
                err = p - yi
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            self.w = [w - lr * g / n for w, g in zip(self.w, gw)]
            self.b -= lr * gb / n
        return self

    def prob(self, xi: Sequence[float]) -> float:
        z = self.b + sum(w * x for w, x in zip(self.w, xi))
        z = max(-60.0, min(60.0, z))
        return 1.0 / (1.0 + math.exp(-z))


def fit_probe_from_records(records, feature_signals: List[str]) -> LogisticProbe:
    """Fit on labeled dev records: features = named confidence signals already in
    the records; label = whether local was correct (score >= 0.5)."""
    X = [[r.confidences[s] for s in feature_signals] for r in records]
    y = [1.0 if r.local_score >= 0.5 else 0.0 for r in records]
    return LogisticProbe().fit(X, y)


class ProbeGate:
    """Wraps a fitted probe over a fixed list of base gates. Its confidence is the
    probe's P(local correct). Build the probe with fit_probe_from_records using
    the same base-gate names, in order."""
    name = "probe"

    def __init__(self, probe: LogisticProbe, base_gates: List[Gate]):
        self.probe = probe
        self.base_gates = base_gates

    def confidence(self, local: LocalOutput, item: Item) -> float:
        feats = compute_confidences(self.base_gates, local, item)
        return self.probe.prob([feats[g.name] for g in self.base_gates])
