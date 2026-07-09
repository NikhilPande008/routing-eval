"""End-to-end: dataset -> mock run -> frontier. Also validates that the harness
correctly rewards a better gate, which is its entire reason to exist.
"""
from routing_eval.datasets import make_standin
from routing_eval.frontier import evaluate
from routing_eval.mock import build_records


def test_pipeline_runs_and_records_wellformed():
    items = make_standin(60, seed=1)
    recs = build_records(items, seed=1)
    assert len(recs) == 180
    r = recs[0]
    assert 0.0 <= r.local_score <= 1.0
    assert {"informative", "random", "anti"}.issubset(r.confidences)
    assert r.remote_total_tokens == r.remote_prompt_tokens + r.remote_completion_tokens


def test_informative_gate_beats_random():
    recs = build_records(make_standin(120, seed=2), calib_noise=0.3, seed=2)
    floor = 0.85
    inf = evaluate(recs, "informative", floor)
    rnd = evaluate(recs, "random", floor)
    assert inf.feasible and rnd.feasible
    # a better-calibrated gate reaches the same floor with fewer remote tokens
    assert inf.operating_point.remote_tokens < rnd.operating_point.remote_tokens
    assert inf.gate_efficiency > rnd.gate_efficiency


def test_informative_gate_beats_anticorrelated():
    recs = build_records(make_standin(120, seed=5), calib_noise=0.3, seed=5)
    floor = 0.85
    inf = evaluate(recs, "informative", floor).gate_efficiency
    anti = evaluate(recs, "anti", floor).gate_efficiency
    assert inf > anti


def test_lower_calibration_noise_improves_efficiency():
    lo = build_records(make_standin(120, seed=3), calib_noise=0.15, seed=3)
    hi = build_records(make_standin(120, seed=3), calib_noise=3.0, seed=3)
    floor = 0.85
    e_lo = evaluate(lo, "informative", floor).gate_efficiency
    e_hi = evaluate(hi, "informative", floor).gate_efficiency
    assert e_lo > e_hi
