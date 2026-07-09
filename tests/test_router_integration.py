"""Prove P2 plugs into P1: stubbed runners -> build_records -> the P1 frontier
tracer, plus the Router escalation policy and its token accounting.
"""
from routing_eval.frontier import evaluate
from routing_eval.gates import FREE_GATES, LogprobGate
from routing_eval.llm import LocalRunner, RemoteRunner, StubClient, stub_response
from routing_eval.router import Router, build_records
from routing_eval.schema import Item

ITEMS = [Item(f"m-{i}", "math", f"{i} plus {i}?", 2 * i, "numeric", {}) for i in range(1, 9)]


def _local_client():
    # local is confident+correct on even i, unconfident+wrong on odd i
    def handler(req):
        q = req["messages"][-1]["content"]
        i = int(q.split()[0])
        if i % 2 == 0:
            return stub_response(["%d" % (2 * i)], token_logprobs=[[-0.02, -0.03]],
                                 usage={"completion_tokens": 1})
        return stub_response(["%d" % (2 * i + 1)], token_logprobs=[[-2.5, -2.8]],
                             usage={"completion_tokens": 1})
    return StubClient(handler)


def _remote_client():
    def handler(req):
        i = int(req["messages"][-1]["content"].split()[0])
        return stub_response(["%d" % (2 * i)],
                             usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
    return StubClient(handler)


def test_build_records_feeds_frontier():
    local = LocalRunner(_local_client(), "local")
    remote = RemoteRunner(_remote_client(), "remote")
    recs = build_records(ITEMS, local, remote, FREE_GATES)
    assert len(recs) == 8
    # local right on the 4 even items, remote right on all -> baselines are known
    fr = evaluate(recs, "logprob", 0.99)
    assert fr.all_local_accuracy == 0.5
    assert fr.all_remote_accuracy == 1.0
    # the logprob gate cleanly separates confident(even) from unconfident(odd),
    # so it should reach 1.0 accuracy by escalating only the 4 odd items
    assert fr.feasible
    assert fr.operating_point.accuracy == 1.0
    assert fr.operating_point.escalation_rate == 0.5
    assert fr.gate_efficiency == 1.0            # escalates exactly the wrong items


def test_router_escalation_and_accounting():
    local = LocalRunner(_local_client(), "local")
    remote = RemoteRunner(_remote_client(), "remote")
    even, odd = ITEMS[1], ITEMS[0]              # i=2 (confident), i=1 (unconfident)

    keep = Router(local, remote, LogprobGate(), tau=0.5).route(even)
    assert keep.escalated is False and keep.remote_tokens == 0

    esc = Router(local, remote, LogprobGate(), tau=0.5).route(odd)
    assert esc.escalated is True and esc.remote_tokens == 12 and esc.answer == "2"

    # tau below every confidence -> never escalate; tau above -> always escalate
    assert Router(local, remote, LogprobGate(), tau=0.0).route(odd).escalated is False
    assert Router(local, remote, LogprobGate(), tau=1.0).route(even).escalated is True
