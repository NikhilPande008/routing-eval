import math

from routing_eval.gates import (DeterministicGate, LogisticProbe, LogprobGate,
                                SelfConsistencyGate)
from routing_eval.llm.runners import LocalOutput
from routing_eval.schema import Item

MATH = Item("m", "math", "2+2?", 4, "numeric", {})
MC = Item("c", "classification", "?", "positive", "multiple_choice", {},
          allowed=["positive", "negative", "neutral"])


def _lo(answer="4", logprobs=None, samples=None):
    return LocalOutput(answer, logprobs or [], samples or [answer], 1, {})


def test_logprob_gate_monotone_and_bounded():
    g = LogprobGate()
    high = g.confidence(_lo(logprobs=[-0.01, -0.02]), MATH)
    low = g.confidence(_lo(logprobs=[-3.0, -2.5]), MATH)
    assert 0.0 < low < high <= 1.0
    assert g.confidence(_lo(logprobs=[]), MATH) == 0.0
    assert math.isclose(g.confidence(_lo(logprobs=[0.0, 0.0]), MATH), 1.0)


def test_self_consistency_fraction():
    g = SelfConsistencyGate()
    assert g.confidence(_lo(samples=["4", "4", "4", "4"]), MATH) == 1.0
    assert g.confidence(_lo(samples=["4", "4", "5"]), MATH) == 2 / 3
    # task-aware bucketing: "4" and "the answer is 4" agree on the number
    assert g.confidence(_lo(samples=["4", "the answer is 4"]), MATH) == 1.0


def test_deterministic_gate_structure_checks():
    g = DeterministicGate()
    assert g.confidence(_lo("42"), MATH) == 1.0
    assert g.confidence(_lo("banana"), MATH) == 0.0          # no number -> escalate
    assert g.confidence(_lo("clearly positive"), MC) == 1.0
    assert g.confidence(_lo("maybe"), MC) == 0.0             # no allowed label


def test_logistic_probe_learns_separable_signal():
    # feature strongly predicts the label; probe should separate the classes
    X = [[0.9], [0.85], [0.8], [0.1], [0.15], [0.2]]
    y = [1, 1, 1, 0, 0, 0]
    p = LogisticProbe().fit(X, y, epochs=800, lr=0.5)
    assert p.prob([0.9]) > 0.5 > p.prob([0.1])
