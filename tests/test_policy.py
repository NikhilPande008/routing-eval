import time

from routing_eval.classify import ClassificationResult
from routing_eval.llm import StubClient, stub_response
from routing_eval.llm.runners import LocalRunner
from routing_eval.modelselect import LocalViability, ModelCategoryRanking
from routing_eval.policy import (DEFAULT_POLICY_PATH, PROMPT_TEMPLATES, PolicyEntry,
                                 PolicyRouter, generate_policy, load_policy,
                                 resolve_entry, save_policy)


class FixedClassifier:
    """Test double: maps a prompt substring to a fixed category, sidestepping
    keyword-heuristic tuning so policy-routing tests aren't coupled to it."""

    def __init__(self, mapping, confidence=1.0):
        self.mapping = mapping
        self.confidence = confidence

    def classify(self, prompt):
        for needle, category in self.mapping.items():
            if needle in prompt:
                return ClassificationResult(category, self.confidence, {})
        return ClassificationResult("uncategorized", 0.0, {})


def _policy(entries):
    entries = dict(entries)
    entries.setdefault("_default", {"tier": "fireworks", "model": None, "max_tokens": 64,
                                    "prompt_template": "default"})
    return {cat: PolicyEntry.from_dict(d) for cat, d in entries.items()}


# ---------------------------------------------------------------------------
# Checked-in default
# ---------------------------------------------------------------------------

def test_checked_in_default_routes_everything_to_fireworks_pinned_to_kimi():
    """Pinned 2026-07-09 (D30/D31): the bake-off (D30) showed kimi-k2p7-code
    beats minimax-m3 on tokens in every category tested. Retry-on-opposite-
    model (D31) is the safety net if kimi ever becomes unavailable at
    grading time -- the pin is no longer "the safe null-model fallback",
    it's an informed choice backed by real data, with its own fallback."""
    policy = load_policy(DEFAULT_POLICY_PATH)
    entry = resolve_entry(policy, "anything")
    assert entry.tier == "fireworks"
    assert entry.model == "kimi-k2p7-code"


def test_resolve_entry_falls_back_through_default_then_safe_fallback():
    policy = _policy({"math": {"tier": "local", "model": None, "max_tokens": 32,
                               "prompt_template": "default"}})
    assert resolve_entry(policy, "math").tier == "local"
    assert resolve_entry(policy, "unseen-category").tier == "fireworks"   # via "_default"
    assert resolve_entry({}, "anything").tier == "fireworks"              # via SAFE_FALLBACK_ENTRY


# ---------------------------------------------------------------------------
# PolicyRouter: local-tier makes zero Fireworks calls
# ---------------------------------------------------------------------------

def test_local_tier_makes_zero_fireworks_calls():
    policy = _policy({"math": {"tier": "local", "model": None, "max_tokens": 32,
                               "prompt_template": "default"}})
    local_client = StubClient(stub_response(["4"]))
    remote_client = StubClient(stub_response(["should-not-be-called"]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw-model"],
                          classifier=FixedClassifier({"2+2": "math"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "2+2?"}, 0)

    assert result == {"task_id": "t1", "answer": "4"}
    assert len(local_client.calls) == 1
    assert len(remote_client.calls) == 0


# ---------------------------------------------------------------------------
# PolicyRouter: fireworks-tier hits the specified model
# ---------------------------------------------------------------------------

def test_fireworks_tier_hits_the_policy_specified_model():
    policy = _policy({"knowledge": {"tier": "fireworks", "model": "specific-model",
                                    "max_tokens": 32, "prompt_template": "default"}})
    remote_client = StubClient(stub_response(["Paris"]))
    router = PolicyRouter(policy, remote_client=remote_client,
                          allowed_models=["default-model", "other-model"],
                          classifier=FixedClassifier({"capital": "knowledge"}), local=None)

    result = router.route_task({"task_id": "t1", "prompt": "capital of France?"}, 0)

    assert result == {"task_id": "t1", "answer": "Paris"}
    assert remote_client.calls[0]["model"] == "accounts/fireworks/models/specific-model"


def test_checked_in_default_routes_code_and_sentiment_categories_to_the_right_template():
    from routing_eval.policy import CODE_ONLY_SYSTEM, SENTIMENT_SYSTEM
    policy = load_policy(DEFAULT_POLICY_PATH)
    assert resolve_entry(policy, "code_debug").prompt_template == "code_only"
    assert resolve_entry(policy, "code_gen").prompt_template == "code_only"
    assert resolve_entry(policy, "sentiment").prompt_template == "sentiment_with_justification"
    # tier/model: all pinned to kimi-k2p7-code (D30's bake-off winner) -- only
    # the prompt template differs by category
    for cat in ("code_debug", "code_gen", "sentiment"):
        entry = resolve_entry(policy, cat)
        assert entry.tier == "fireworks" and entry.model == "kimi-k2p7-code"
    assert PROMPT_TEMPLATES["code_only"] == CODE_ONLY_SYSTEM
    assert PROMPT_TEMPLATES["sentiment_with_justification"] == SENTIMENT_SYSTEM


def test_code_only_template_sends_the_no_fence_instruction():
    policy = load_policy(DEFAULT_POLICY_PATH)
    remote_client = StubClient(stub_response(["def f(): return 1"]))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m"],
                          classifier=FixedClassifier({"bug": "code_debug"}), local=None)

    router.route_task({"task_id": "t1", "prompt": "fix this bug"}, 0)

    system_msg = remote_client.calls[0]["messages"][0]
    assert system_msg["role"] == "system"
    assert "no markdown code fences" in system_msg["content"]


def test_code_only_strips_a_markdown_fence_even_if_the_model_adds_one():
    policy = load_policy(DEFAULT_POLICY_PATH)
    fenced = "```python\ndef f():\n    return 1\n```"
    remote_client = StubClient(stub_response([fenced]))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m"],
                          classifier=FixedClassifier({"bug": "code_debug"}), local=None)

    result = router.route_task({"task_id": "t1", "prompt": "fix this bug"}, 0)

    assert result["answer"] == "def f():\n    return 1"
    assert "```" not in result["answer"]


def test_sentiment_template_sends_the_justification_instruction():
    policy = load_policy(DEFAULT_POLICY_PATH)
    remote_client = StubClient(stub_response(["Positive: great battery life."]))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m"],
                          classifier=FixedClassifier({"sentiment": "sentiment"}), local=None)

    router.route_task({"task_id": "t1", "prompt": "sentiment of this review"}, 0)

    system_msg = remote_client.calls[0]["messages"][0]
    assert "justification" in system_msg["content"]


def test_strip_code_fence_passes_through_unfenced_text():
    from routing_eval.policy import _strip_code_fence
    assert _strip_code_fence("def f(): return 1") == "def f(): return 1"


def test_fireworks_call_logs_token_usage(capsys):
    policy = _policy({"knowledge": {"tier": "fireworks", "model": "specific-model",
                                    "max_tokens": 32, "prompt_template": "default"}})
    remote_client = StubClient(stub_response(
        ["Paris"], usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["default-model"],
                          classifier=FixedClassifier({"capital": "knowledge"}), local=None)

    router.route_task({"task_id": "t1", "prompt": "capital of France?"}, 0)

    err = capsys.readouterr().err
    assert "prompt=12" in err and "completion=3" in err and "total=15" in err


def test_fireworks_tier_with_null_model_uses_first_allowed_model():
    policy = _policy({})   # only "_default": model=None
    remote_client = StubClient(stub_response(["ok"]))
    router = PolicyRouter(policy, remote_client=remote_client,
                          allowed_models=["first-model", "second-model"],
                          classifier=FixedClassifier({}), local=None)

    router.route_task({"task_id": "t1", "prompt": "anything"}, 0)

    assert remote_client.calls[0]["model"] == "first-model"


# ---------------------------------------------------------------------------
# PolicyRouter: retry-on-opposite-model before empty is the last resort
# ---------------------------------------------------------------------------

def test_blank_answer_retries_with_a_different_allowed_model():
    policy = _policy({})

    def handler(kw):
        if kw["model"] == "kimi-k2p7-code":
            return stub_response([""])          # blank -- should trigger a retry
        return stub_response(["a real answer"])

    remote_client = StubClient(handler)
    router = PolicyRouter(policy, remote_client=remote_client,
                          allowed_models=["kimi-k2p7-code", "minimax-m3"],
                          classifier=FixedClassifier({}), local=None)

    result = router.route_task({"task_id": "t1", "prompt": "anything"}, 0)

    assert result == {"task_id": "t1", "answer": "a real answer"}
    assert [c["model"] for c in remote_client.calls] == ["kimi-k2p7-code", "minimax-m3"]


def test_exception_on_primary_model_also_retries_with_a_different_model():
    def handler(kw):
        if kw["model"] == "kimi-k2p7-code":
            raise RuntimeError("simulated Fireworks failure")
        return stub_response(["recovered"])

    policy = _policy({})
    remote_client = StubClient(handler)
    router = PolicyRouter(policy, remote_client=remote_client,
                          allowed_models=["kimi-k2p7-code", "minimax-m3"],
                          classifier=FixedClassifier({}), local=None)

    result = router.route_task({"task_id": "t1", "prompt": "anything"}, 0)

    assert result == {"task_id": "t1", "answer": "recovered"}


def test_empty_answer_is_the_last_resort_not_the_first_fallback(capsys):
    """Both models blank -> empty answer, but only after a retry was
    actually attempted -- proves empty isn't returned on the first failure."""
    policy = _policy({})
    remote_client = StubClient(stub_response([""]))
    router = PolicyRouter(policy, remote_client=remote_client,
                          allowed_models=["kimi-k2p7-code", "minimax-m3"],
                          classifier=FixedClassifier({}), local=None)

    result = router.route_task({"task_id": "t1", "prompt": "anything"}, 0)

    assert result == {"task_id": "t1", "answer": ""}
    assert [c["model"] for c in remote_client.calls] == ["kimi-k2p7-code", "minimax-m3"]
    assert "still blank after retry" in capsys.readouterr().err


def test_no_retry_possible_with_only_one_allowed_model(capsys):
    policy = _policy({})
    remote_client = StubClient(stub_response([""]))
    router = PolicyRouter(policy, remote_client=remote_client,
                          allowed_models=["only-model"],
                          classifier=FixedClassifier({}), local=None)

    result = router.route_task({"task_id": "t1", "prompt": "anything"}, 0)

    assert result == {"task_id": "t1", "answer": ""}
    assert len(remote_client.calls) == 1   # no retry call made -- nothing to retry with
    assert "no alternate model" in capsys.readouterr().err


def test_retry_uses_a_bounded_timeout_matching_d19_budget():
    """Each attempt (primary + retry) is capped at D19's 30s/request budget
    by default -- two attempts still comfortably fit inside the 10-minute
    total even in a worst-case all-tasks-need-a-retry scenario."""
    captured_timeouts = []

    def handler(kw):
        return stub_response(["ok"])

    class TimeoutSpyClient(StubClient):
        def chat(self, **kwargs):
            captured_timeouts.append(kwargs.get("timeout"))
            return super().chat(**kwargs)

    policy = _policy({})
    remote_client = TimeoutSpyClient(handler)
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m1", "m2"],
                          classifier=FixedClassifier({}), local=None)

    router.route_task({"task_id": "t1", "prompt": "anything"}, 0)

    assert captured_timeouts == [30.0]   # policy.py's DEFAULT_TIMEOUT_S


# ---------------------------------------------------------------------------
# PolicyRouter: timeout-and-fallback
# ---------------------------------------------------------------------------

def test_slow_local_call_times_out_and_falls_back_to_fireworks():
    policy = _policy({"math": {"tier": "local", "model": None, "max_tokens": 32,
                               "prompt_template": "default", "timeout_s": 0.02}})

    def slow_handler(kw):
        time.sleep(0.2)
        return stub_response(["too-slow"])

    local_client = StubClient(slow_handler)
    remote_client = StubClient(stub_response(["fireworks-fallback-answer"]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw-model"],
                          classifier=FixedClassifier({"2+2": "math"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "2+2?"}, 0)

    assert result == {"task_id": "t1", "answer": "fireworks-fallback-answer"}
    assert len(remote_client.calls) == 1
    router.close()


def test_local_call_failure_falls_back_to_fireworks():
    policy = _policy({"math": {"tier": "local", "model": None, "max_tokens": 32,
                               "prompt_template": "default"}})

    def raising_handler(kw):
        raise RuntimeError("local server unreachable")

    local = LocalRunner(StubClient(raising_handler), model="local-stub")
    remote_client = StubClient(stub_response(["fallback"]))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw-model"],
                          classifier=FixedClassifier({"2+2": "math"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "2+2?"}, 0)

    assert result == {"task_id": "t1", "answer": "fallback"}
    router.close()


# ---------------------------------------------------------------------------
# Low-confidence flag is logged (debuggability requirement)
# ---------------------------------------------------------------------------

def test_low_confidence_classification_is_flagged_in_logs(capsys):
    policy = _policy({})
    remote_client = StubClient(stub_response(["ok"]))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m"],
                          classifier=FixedClassifier({"x": "math"}, confidence=0.1), local=None,
                          low_confidence_threshold=0.5)

    router.route_task({"task_id": "t1", "prompt": "x marks the spot"}, 0)

    err = capsys.readouterr().err
    assert "LOW-CONFIDENCE" in err
    assert "t1" in err


def test_high_confidence_classification_is_not_flagged(capsys):
    policy = _policy({})
    remote_client = StubClient(stub_response(["ok"]))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m"],
                          classifier=FixedClassifier({"x": "math"}, confidence=0.9), local=None,
                          low_confidence_threshold=0.5)

    router.route_task({"task_id": "t1", "prompt": "x marks the spot"}, 0)

    err = capsys.readouterr().err
    assert "LOW-CONFIDENCE" not in err


# ---------------------------------------------------------------------------
# route_all processes every task and always closes the executor
# ---------------------------------------------------------------------------

def test_route_all_processes_every_task():
    policy = _policy({})
    remote_client = StubClient(stub_response(["ok"]))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m"],
                          classifier=FixedClassifier({}), local=None)

    results = router.route_all([{"task_id": "a", "prompt": "p1"}, {"task_id": "b", "prompt": "p2"}])

    assert [r["task_id"] for r in results] == ["a", "b"]


# ---------------------------------------------------------------------------
# generate_policy: cheapest floor-clearing model, honest null when nothing clears
# ---------------------------------------------------------------------------

def test_generate_policy_routes_local_viable_categories_to_local():
    viability = {"math": LocalViability("math", 5, 0.9, 0.5, 1.0, True, "fast + likely-correct")}
    policy = generate_policy(viability, {})
    assert policy["math"]["tier"] == "local"
    assert policy["math"]["model"] is None
    assert policy["_default"]["tier"] == "fireworks"


def test_generate_policy_picks_cheapest_floor_clearing_model():
    ranking = {"knowledge": [
        ModelCategoryRanking("knowledge", "cheap-good", 0.9, 6, 6.0, True),
        ModelCategoryRanking("knowledge", "expensive-good", 0.95, 60, 60.0, True),
        ModelCategoryRanking("knowledge", "bad", 0.1, 4, 4.0, False),
    ]}
    policy = generate_policy({}, ranking)
    assert policy["knowledge"]["tier"] == "fireworks"
    assert policy["knowledge"]["model"] == "cheap-good"


def test_generate_policy_leaves_model_null_when_nothing_clears_the_floor(capsys):
    ranking = {"hard-category": [ModelCategoryRanking("hard-category", "bad", 0.1, 4, 4.0, False)]}
    policy = generate_policy({}, ranking)
    assert policy["hard-category"]["tier"] == "fireworks"
    assert policy["hard-category"]["model"] is None
    assert "no model clears" in capsys.readouterr().err


def test_generate_policy_prefers_local_over_a_clearing_fireworks_model():
    viability = {"math": LocalViability("math", 5, 0.95, 0.3, 0.5, True, "fast + likely-correct")}
    ranking = {"math": [ModelCategoryRanking("math", "cheap-good", 0.9, 6, 6.0, True)]}
    policy = generate_policy(viability, ranking)
    assert policy["math"]["tier"] == "local"   # zero tokens beats any Fireworks model


def test_save_and_load_policy_round_trip(tmp_path):
    viability = {"math": LocalViability("math", 5, 0.9, 0.5, 1.0, True, "fast + likely-correct")}
    policy_dict = generate_policy(viability, {})
    path = tmp_path / "draft.json"
    save_policy(policy_dict, str(path))

    loaded = load_policy(str(path))
    assert loaded["math"].tier == "local"
    assert loaded["_default"].tier == "fireworks"
