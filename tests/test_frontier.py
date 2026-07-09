"""Frontier logic, verified against a 4-item case computed entirely by hand.

Items (score_local, score_remote, cost, conf):
  A (1, 1, 10, 0.9)   local right, remote right
  B (1, 0, 10, 0.8)   local right, remote WRONG  -> escalating HURTS
  C (0, 1,  5, 0.2)   local wrong, remote right, cheap
  D (0, 1, 20, 0.1)   local wrong, remote right, expensive

Escalation order by ascending confidence is D, C, B, A. Everything below is
derived from those four rows.
"""
from routing_eval.frontier import evaluate, oracle_tokens, trace_frontier
from routing_eval.schema import Record


def _rec(rid, ls, rs, cost, conf):
    return Record(rid, "t", "unknown", None, "la", ls, cost,
                  "ra", rs, cost, 0, cost, {"c": conf}, None)


def handcrafted():
    return [_rec("A", 1.0, 1.0, 10, 0.9),
            _rec("B", 1.0, 0.0, 10, 0.8),
            _rec("C", 0.0, 1.0, 5, 0.2),
            _rec("D", 0.0, 1.0, 20, 0.1)]


def test_baselines_and_ceiling():
    fr = trace_frontier(handcrafted(), "c")
    assert fr.all_local_accuracy == 0.5           # A,B right locally
    assert fr.all_remote_accuracy == 0.75         # A,C,D right remotely (B wrong)
    assert fr.all_remote_tokens == 45             # 10+10+5+20
    assert fr.union_ceiling == 1.0                # every item is right on some model


def test_frontier_points_including_nonmonotonic():
    fr = trace_frontier(handcrafted(), "c")
    pts = {(p.remote_tokens, round(p.accuracy, 3)) for p in fr.points}
    assert (0, 0.5) in pts                         # escalate none
    assert (20, 0.75) in pts                       # escalate D
    assert (25, 1.0) in pts                        # escalate D,C
    assert (35, 0.75) in pts                       # escalate D,C,B -> B drags acc back down
    assert (45, 0.75) in pts                       # escalate all
    toks = [p.remote_tokens for p in fr.points]
    assert toks == sorted(toks)                    # tokens monotonic along the curve


def test_operating_point_and_oracle_floor_075():
    fr = evaluate(handcrafted(), "c", 0.75)
    assert fr.feasible
    assert fr.operating_point.remote_tokens == 20  # gate escalates D (lowest conf) first
    assert fr.oracle_tokens == 5                   # oracle escalates C (cheapest helpful item)
    assert fr.oracle_exact is True
    assert abs(fr.gate_efficiency - 0.25) < 1e-9   # 5 / 20


def test_floor_100_gate_matches_oracle():
    fr = evaluate(handcrafted(), "c", 1.0)
    # to reach 1.0 you must escalate both C and D; the gate's two lowest-conf
    # items are exactly C and D, so the gate is optimal here.
    assert fr.operating_point.remote_tokens == 25
    assert fr.oracle_tokens == 25
    assert fr.gate_efficiency == 1.0


def test_infeasible_floor_flagged():
    fr = evaluate(handcrafted(), "c", 1.01)
    assert fr.feasible is False
    assert fr.operating_point is None
    assert oracle_tokens(handcrafted(), 1.01)[0] is None


def test_zero_tokens_when_local_already_meets_floor():
    fr = evaluate(handcrafted(), "c", 0.4)
    assert fr.operating_point.remote_tokens == 0
    assert fr.oracle_tokens == 0
    assert fr.gate_efficiency == 1.0
