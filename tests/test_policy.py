import time

from routing_eval.classify import ClassificationResult, TieredClassifier
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

def test_checked_in_default_is_local_first_with_kimi_escalation():
    """D52 (all-in redesign): the default/general bucket is LOCAL-first with
    two-sample self-consistency, escalating to kimi-k2p7-code (still pinned,
    D30) on any validator/agreement failure, timeout, or governor
    exhaustion. The remote escalation path is unchanged from the
    live-validated configs."""
    policy = load_policy(DEFAULT_POLICY_PATH)
    entry = resolve_entry(policy, "anything")
    assert entry.tier == "local"
    assert entry.model == "kimi-k2p7-code"          # the escalation target
    assert entry.local_prompt_template == "knowledge_local"
    assert entry.local_n_samples == 2               # self-consistency fence
    assert entry.prompt_template == "default"       # remote escalation treatment


def test_policy_router_defaults_to_tiered_classifier():
    """D41: PolicyRouter's default classifier is TieredClassifier (D37's
    code/sentiment detectors + entity/math/logic token-tier detectors) --
    nothing in conformance.py/cli.py's `score` command passes a classifier
    explicitly, so this default IS the deployed behavior."""
    policy = load_policy(DEFAULT_POLICY_PATH)
    router = PolicyRouter(policy, remote_client=StubClient(stub_response(["ok"])),
                          allowed_models=["m"])
    assert isinstance(router.classifier, TieredClassifier)


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
    # "code" is what TwoWayClassifier (D37, the deployed default) actually
    # emits; "code_debug"/"code_gen" are kept only for KeywordClassifier-based
    # tooling (bakeoff/generate-policy) -- all three must still resolve to
    # the same code_only template.
    assert resolve_entry(policy, "code").prompt_template == "code_only"
    assert resolve_entry(policy, "code_debug").prompt_template == "code_only"
    assert resolve_entry(policy, "code_gen").prompt_template == "code_only"
    assert resolve_entry(policy, "sentiment").prompt_template == "sentiment_with_justification"
    # D48 ladder (PLAN-TOKEN-OPT.md): sentiment is the first local-tier rung
    # of the redesigned ladder (10s timeout, speed probe, validator-gated,
    # escalation on the ORIGINAL remote template). Everything else stays
    # remote until its own rung is live-validated. kimi stays pinned (D30)
    # so escalation never depends on ALLOWED_MODELS ordering.
    # D53: code is local-first with EXECUTION PROOF (kept only when the task
    # carries parseable examples and the generated code reproduces them);
    # logic is local-first with n=2 prompt-diversity agreement. Escalation
    # stays kimi-pinned on the original remote templates.
    for cat in ("code", "code_debug", "code_gen"):
        entry = resolve_entry(policy, cat)
        assert entry.tier == "local" and entry.model == "kimi-k2p7-code"
        assert entry.local_prompt_template == "code_local"
        assert entry.prompt_template == "code_only"     # escalation treatment
    # logic went local in a D53 draft and was REVERTED after the judge-proxy
    # caught agreement-by-chance on a wrong puzzle answer (tiny answer
    # spaces): logic stays remote on its live-validated template.
    logic = resolve_entry(policy, "logic")
    assert logic.tier == "fireworks"
    assert logic.prompt_template == "logic_answer_last"
    # D52: per-task timeouts are GENEROUS by design (the batch-level
    # local_budget_s governor bounds total spend; slow hardware exhausts the
    # budget and the remainder goes remote -- the old tight caps just turned
    # slow hardware into always-escalate with zero savings). math/knowledge
    # additionally require two-sample agreement (local_n_samples=2).
    for cat, local_tmpl, t, n in (("sentiment", "sentiment_local", 20, 1),
                                  ("entity_extraction", "entity_local", 30, 1),
                                  ("summarization", "summarize_exact", 40, 1),
                                  ("math", "math_local", 60, 2)):
        entry = resolve_entry(policy, cat)
        assert entry.tier == "local" and entry.model == "kimi-k2p7-code"
        assert entry.local_prompt_template == local_tmpl
        assert entry.timeout_s == t
        assert entry.local_n_samples == n
    # Fireworks ESCALATION treatments stay on the live-validated remote
    # templates -- only the local calls use the *_local prompts.
    assert resolve_entry(policy, "summarization").prompt_template == "default"
    assert resolve_entry(policy, "math").prompt_template == "math_direct"
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


# ---------------------------------------------------------------------------
# D40: kill the anti-explanation prompt, eliminate truncation as a failure mode
# ---------------------------------------------------------------------------

def test_default_template_is_accurate_not_terse():
    """D40 killed the terse anti-explanation prompt after submission #2
    failed at 63.2%; D41 then TRIMMED the replacement's wording (prompt
    overhead was 81% of token spend) without reintroducing the old failure:
    the trimmed default must still ask for real explanations, respect stated
    length constraints, and answer every part -- and the D40-banned
    brevity-above-all phrases must stay gone."""
    from routing_eval.policy import PROMPT_TEMPLATES
    default = PROMPT_TEMPLATES["default"]
    assert "as briefly as possible" not in default
    assert "do not explain" not in default.lower()
    assert "2-5 sentences" in default
    assert "every part" in default
    assert "follow it exactly" in default


def test_math_routes_to_the_isolated_step2a_trim():
    """D45/Step 2a: math is the ONE category trimmed this round, deliberately
    isolated (the D42 lesson, reinforced by D45's local-tier failure -- never
    bundle multiple category changes into one push). Keeps the
    "final answer, then one line of working" shape every prior live-
    validated math treatment shared; retired templates must stay gone."""
    from routing_eval.policy import PROMPT_TEMPLATES
    from routing_eval.prompts import MATH_DIRECT_SYSTEM
    policy = load_policy(DEFAULT_POLICY_PATH)
    assert resolve_entry(policy, "math").prompt_template == "math_direct"
    assert PROMPT_TEMPLATES["math_direct"] == MATH_DIRECT_SYSTEM
    assert "final answer" in MATH_DIRECT_SYSTEM.lower()
    assert "one line" in MATH_DIRECT_SYSTEM.lower()
    for retired in ("math_minimal", "entity_list", "logic_minimal"):
        assert retired not in PROMPT_TEMPLATES


def test_logic_routes_to_the_answer_last_template_not_accurate_full():
    """D43: authorized deviation from D42's conservatism. The Sam/Jo/Lee
    logic puzzle went 3/3 reproducible under ACCURATE_FULL_SYSTEM's
    answer-first wording; this deduction-first/answer-last template has 6/6
    live A/B validation from D41 (including 3/3 on the exact flaky prompt)
    and was never itself implicated in the D41 trim's live failure."""
    from routing_eval.policy import PROMPT_TEMPLATES
    from routing_eval.prompts import LOGIC_ANSWER_LAST_SYSTEM
    policy = load_policy(DEFAULT_POLICY_PATH)
    assert resolve_entry(policy, "logic").prompt_template == "logic_answer_last"
    assert PROMPT_TEMPLATES["logic_answer_last"] == LOGIC_ANSWER_LAST_SYSTEM
    assert "Answer:" in LOGIC_ANSWER_LAST_SYSTEM
    assert LOGIC_ANSWER_LAST_SYSTEM.index("deduction") < LOGIC_ANSWER_LAST_SYSTEM.index("Answer:")


def test_entity_extraction_routes_to_the_explicit_type_label_template():
    """D42: entity extraction is restored from the failed 14-token one-liner
    to a template that names the type labels explicitly and requires every
    entity -- the shape the 17/19 run's entity answers actually had."""
    from routing_eval.policy import PROMPT_TEMPLATES
    policy = load_policy(DEFAULT_POLICY_PATH)
    assert resolve_entry(policy, "entity_extraction").prompt_template == "entity_typed"
    entity = PROMPT_TEMPLATES["entity_typed"]
    for label in ("PERSON", "ORGANIZATION", "LOCATION", "DATE"):
        assert label in entity
    assert "every entity" in entity


def test_knowledge_and_summarization_still_get_the_fuller_default_template():
    """D41's tiering must NOT terse-ify the two categories kept on fuller
    treatment -- knowledge/summarization have no policy entry, so they fall
    through '_default'. Guard the fall-through explicitly."""
    policy = load_policy(DEFAULT_POLICY_PATH)
    assert resolve_entry(policy, "knowledge").prompt_template == "default"
    assert resolve_entry(policy, "summarization").prompt_template == "default"
    assert resolve_entry(policy, "general").prompt_template == "default"


def test_checked_in_default_max_tokens_is_512_not_256():
    policy = load_policy(DEFAULT_POLICY_PATH)
    for cat in ("code", "code_debug", "code_gen", "sentiment", "_default"):
        assert resolve_entry(policy, cat).max_tokens == 512


def test_policy_entry_defaults_to_512_max_tokens():
    assert PolicyEntry(tier="fireworks").max_tokens == 512
    assert PolicyEntry.from_dict({"tier": "fireworks"}).max_tokens == 512


def test_truncated_answer_retries_once_with_doubled_max_tokens():
    """A finish_reason='length' response is a guaranteed judge failure (D18)
    -- one same-model retry with 2x the cap should recover it."""
    policy = _policy({"knowledge": {"tier": "fireworks", "model": "m", "max_tokens": 100,
                                    "prompt_template": "default"}})

    def handler(kw):
        if kw["max_tokens"] == 100:
            return stub_response(["a truncated ans"], finish_reason="length")
        return stub_response(["a complete, untruncated answer."], finish_reason="stop")

    remote_client = StubClient(handler)
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m"],
                          classifier=FixedClassifier({"capital": "knowledge"}), local=None)

    result = router.route_task({"task_id": "t1", "prompt": "capital of France?"}, 0)

    assert result == {"task_id": "t1", "answer": "a complete, untruncated answer."}
    assert [c["max_tokens"] for c in remote_client.calls] == [100, 200]


def test_length_retry_happens_at_most_once_per_model():
    """Still truncated after the doubled-cap retry -> accept it (truncated
    but non-blank), do not double a third time. The outer blank-retry
    (D31, a different model) is the next safety net, not another doubling."""
    policy = _policy({"knowledge": {"tier": "fireworks", "model": "m", "max_tokens": 100,
                                    "prompt_template": "default"}})
    remote_client = StubClient(stub_response(["still truncated"], finish_reason="length"))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m"],
                          classifier=FixedClassifier({"capital": "knowledge"}), local=None)

    result = router.route_task({"task_id": "t1", "prompt": "capital of France?"}, 0)

    assert result == {"task_id": "t1", "answer": "still truncated"}
    assert [c["max_tokens"] for c in remote_client.calls] == [100, 200]   # exactly one retry


def test_finish_reason_is_logged_on_every_call(capsys):
    policy = _policy({"knowledge": {"tier": "fireworks", "model": "m", "max_tokens": 100,
                                    "prompt_template": "default"}})
    remote_client = StubClient(stub_response(["Paris"]))
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["m"],
                          classifier=FixedClassifier({"capital": "knowledge"}), local=None)

    router.route_task({"task_id": "t1", "prompt": "capital of France?"}, 0)

    assert "finish_reason='stop'" in capsys.readouterr().err


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


# ---------------------------------------------------------------------------
# Local tier: validation gate (2026-07-11) -- a local answer is kept only if
# localcheck passes it; rejection escalates to Fireworks.
# ---------------------------------------------------------------------------

def test_local_answer_failing_validation_escalates_to_fireworks():
    policy = _policy({"sentiment": {"tier": "local", "model": None, "max_tokens": 64,
                                    "prompt_template": "sentiment_with_justification"}})
    # no sentiment label at all -> localcheck must reject
    local_client = StubClient(stub_response(["The reviewer has feelings."]))
    remote_client = StubClient(stub_response(["Mixed: praises battery, dislikes screen."]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw-model"],
                          classifier=FixedClassifier({"sentiment": "sentiment"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "sentiment of this review?"}, 0)

    assert result["answer"] == "Mixed: praises battery, dislikes screen."
    assert len(local_client.calls) == 2   # original + D54 free retry, both rejected
    assert len(remote_client.calls) == 1


def test_local_answer_passing_validation_is_kept_and_remote_untouched():
    policy = _policy({"sentiment": {"tier": "local", "model": None, "max_tokens": 64,
                                    "prompt_template": "sentiment_with_justification"}})
    local_client = StubClient(stub_response(["Positive: praises the battery life."]))
    remote_client = StubClient(stub_response(["should-not-be-called"]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw-model"],
                          classifier=FixedClassifier({"sentiment": "sentiment"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "sentiment of this review?"}, 0)

    assert result["answer"] == "Positive: praises the battery life."
    assert len(remote_client.calls) == 0


def test_blank_local_answer_escalates_to_fireworks():
    policy = _policy({"math": {"tier": "local", "model": None, "max_tokens": 32,
                               "prompt_template": "default"}})
    local_client = StubClient(stub_response([""]))
    remote_client = StubClient(stub_response(["4"]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw-model"],
                          classifier=FixedClassifier({"2+2": "math"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "2+2?"}, 0)

    assert result["answer"] == "4"
    assert len(remote_client.calls) == 1


def test_local_call_uses_entry_template_not_terse_default():
    """The local call must carry the policy entry's template as its system
    prompt -- LocalRunner's own default (None) would fall back to the terse
    runner-level DEFAULT_SYSTEM, exactly the D40 failure mode."""
    from routing_eval.prompts import SENTIMENT_SYSTEM
    policy = _policy({"sentiment": {"tier": "local", "model": None, "max_tokens": 64,
                                    "prompt_template": "sentiment_with_justification"}})
    local_client = StubClient(stub_response(["Positive: praises the battery life."]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=StubClient(stub_response(["x"])),
                          allowed_models=["fw-model"],
                          classifier=FixedClassifier({"sentiment": "sentiment"}), local=local)

    router.route_task({"task_id": "t1", "prompt": "sentiment of this review?"}, 0)

    assert local_client.calls[0]["messages"][0] == {"role": "system",
                                                    "content": SENTIMENT_SYSTEM}


def test_local_prompt_template_overrides_local_call_only():
    """summarization: the LOCAL call uses summarize_exact while a rejected
    answer's Fireworks escalation keeps the live-validated 'default' remote
    treatment -- the whole point of local_prompt_template."""
    from routing_eval.prompts import ACCURATE_GENERIC_SYSTEM, SUMMARIZE_SYSTEM
    policy = _policy({"summarization": {"tier": "local", "model": None, "max_tokens": 64,
                                        "prompt_template": "default",
                                        "local_prompt_template": "summarize_exact"}})
    local_client = StubClient(stub_response(["I'm sorry, I cannot summarize this."]))
    remote_client = StubClient(stub_response(["The council approved the budget."]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw-model"],
                          classifier=FixedClassifier({"Summarize": "summarization"}),
                          local=local)

    result = router.route_task(
        {"task_id": "t1", "prompt": "Summarize: the council approved the budget."}, 0)

    assert result["answer"] == "The council approved the budget."
    assert local_client.calls[0]["messages"][0]["content"] == SUMMARIZE_SYSTEM
    assert remote_client.calls[0]["messages"][0]["content"] == ACCURATE_GENERIC_SYSTEM


# ---------------------------------------------------------------------------
# D52: batch-level local time governor + two-sample self-consistency
# ---------------------------------------------------------------------------

def test_local_budget_exhaustion_skips_local_and_goes_remote():
    policy = _policy({"knowledge": {"tier": "local", "model": None, "max_tokens": 64,
                                    "prompt_template": "default"}})
    local_client = StubClient(stub_response(["should-not-be-used"]))
    remote_client = StubClient(stub_response(["Paris"]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw"],
                          classifier=FixedClassifier({"capital": "knowledge"}),
                          local=local, local_budget_s=100.0)
    router._local_spent_s = 100.0   # budget already gone

    result = router.route_task({"task_id": "t1", "prompt": "capital of France?"}, 0)

    assert result["answer"] == "Paris"
    assert len(local_client.calls) == 0
    assert len(remote_client.calls) == 1


def test_two_sample_agreement_keeps_first_sample():
    policy = _policy({"math": {"tier": "local", "model": None, "max_tokens": 64,
                               "prompt_template": "default", "local_n_samples": 2}})
    answers = iter(["Working: 2+2.\nAnswer: 4", "It is 2+2.\nAnswer: 4"])
    local_client = StubClient(lambda kw: stub_response([next(answers)]))
    remote_client = StubClient(stub_response(["should-not-be-called"]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw"],
                          classifier=FixedClassifier({"2+2": "math"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "2+2?"}, 0)

    assert result["answer"] == "Working: 2+2.\nAnswer: 4"   # first (temp-0) sample
    assert len(local_client.calls) == 2
    assert local_client.calls[1]["temperature"] == 0.7      # diversity sample
    assert len(remote_client.calls) == 0


def test_two_sample_disagreement_escalates():
    policy = _policy({"math": {"tier": "local", "model": None, "max_tokens": 64,
                               "prompt_template": "default", "local_n_samples": 2}})
    answers = iter(["Answer: 4", "Answer: 5",           # disagree
                    "Answer: 4", "Answer: 5"])           # D54 retry disagrees too
    local_client = StubClient(lambda kw: stub_response([next(answers)]))
    remote_client = StubClient(stub_response(["4"]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw"],
                          classifier=FixedClassifier({"2+2": "math"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "2+2?"}, 0)

    assert result["answer"] == "4"                          # kimi's answer
    assert len(local_client.calls) == 4                     # 2 samples + retry pair
    assert len(remote_client.calls) == 1


# ---------------------------------------------------------------------------
# D54: one free local format-retry before escalation
# ---------------------------------------------------------------------------

def test_local_retry_recovers_format_failure():
    policy = _policy({"sentiment": {"tier": "local", "model": None, "max_tokens": 64,
                                    "prompt_template": "sentiment_with_justification"}})
    answers = iter(["The reviewer has feelings.",                       # no label -> rejected
                    "Positive: praises the battery life sincerely."])   # retry passes
    local_client = StubClient(lambda kw: stub_response([next(answers)]))
    remote_client = StubClient(stub_response(["should-not-be-called"]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw"],
                          classifier=FixedClassifier({"sentiment": "sentiment"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "sentiment of this?"}, 0)

    assert result["answer"] == "Positive: praises the battery life sincerely."
    assert len(local_client.calls) == 2
    assert "rejected because" in local_client.calls[1]["messages"][0]["content"]
    assert len(remote_client.calls) == 0


def test_local_retry_failure_still_escalates():
    policy = _policy({"sentiment": {"tier": "local", "model": None, "max_tokens": 64,
                                    "prompt_template": "sentiment_with_justification"}})
    local_client = StubClient(stub_response(["still no label here at all"]))
    remote_client = StubClient(stub_response(["Positive: great."]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw"],
                          classifier=FixedClassifier({"sentiment": "sentiment"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "sentiment of this?"}, 0)

    assert result["answer"] == "Positive: great."
    assert len(local_client.calls) == 2      # original + retry, both rejected
    assert len(remote_client.calls) == 1


def test_local_retry_with_consistency_recheck():
    """n>=2 category: a retry that passes the validator must ALSO pass a fresh
    agreement check before being kept."""
    policy = _policy({"math": {"tier": "local", "model": None, "max_tokens": 64,
                               "prompt_template": "default", "local_n_samples": 2}})
    answers = iter(["no answer line",          # sample 1: validator ok (generic), but...
                    "Answer: 7",               # sample 2: agreement fails (no line in s1)
                    "Answer: 7",               # retry: has the line
                    "Answer: 7"])              # fresh diversity sample agrees
    local_client = StubClient(lambda kw: stub_response([next(answers)]))
    remote_client = StubClient(stub_response(["should-not-be-called"]))
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=remote_client, allowed_models=["fw"],
                          classifier=FixedClassifier({"3+4": "math"}), local=local)

    result = router.route_task({"task_id": "t1", "prompt": "3+4?"}, 0)

    assert result["answer"] == "Answer: 7"
    assert len(local_client.calls) == 4
    assert len(remote_client.calls) == 0


# ---------------------------------------------------------------------------
# D55: cheapest-first local processing order (output order preserved)
# ---------------------------------------------------------------------------

def test_route_all_processes_cheap_locals_first_but_preserves_output_order():
    policy = _policy({
        "sentiment": {"tier": "local", "model": None, "max_tokens": 64,
                      "prompt_template": "sentiment_with_justification"},
        "math": {"tier": "local", "model": None, "max_tokens": 64,
                 "prompt_template": "default"},
    })
    seen = []
    def handler(kw):
        seen.append(kw["messages"][-1]["content"])
        if "sentiment" in kw["messages"][-1]["content"]:
            return stub_response(["Positive: praises it sincerely."])
        return stub_response(["Answer: 4"])
    local_client = StubClient(handler)
    local = LocalRunner(local_client, model="local-stub")
    router = PolicyRouter(policy, remote_client=StubClient(stub_response(["x"])),
                          allowed_models=["fw"],
                          classifier=FixedClassifier({"2+2": "math",
                                                      "sentiment": "sentiment"}),
                          local=local)

    tasks = [{"task_id": "t-math", "prompt": "2+2?"},
             {"task_id": "t-sent", "prompt": "sentiment of this?"}]
    results = router.route_all(tasks)

    # output order matches input order...
    assert [r["task_id"] for r in results] == ["t-math", "t-sent"]
    # ...but the sentiment task was PROCESSED first (cheapest local category)
    assert "sentiment" in seen[0]
