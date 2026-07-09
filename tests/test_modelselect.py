import json

from routing_eval.llm import StubClient, stub_response
from routing_eval.llm.runners import LocalRunner
from routing_eval.modelselect import (TaskResult, assess_local_viability,
                                      evaluate_by_category, load_bakeoff_records,
                                      load_task_results, rank_models_by_category,
                                      run_bakeoff, run_local_probe,
                                      run_model_over_tasks, save_bakeoff_records,
                                      save_task_results)

TASKS = [
    {"task_id": "m1", "category": "math", "prompt": "2+2?", "gold": "4", "scorer": "exact"},
    {"task_id": "m2", "category": "math", "prompt": "10-3?", "gold": "7", "scorer": "exact"},
    {"task_id": "k1", "category": "knowledge", "prompt": "capital of France?",
     "gold": "Paris", "scorer": "exact"},
]


def test_evaluate_by_category_rolls_up_accuracy_and_tokens():
    results = [
        TaskResult("m1", "math", "4", 1.0, 0, 2, 2, latency_s=0.1),
        TaskResult("m2", "math", "wrong", 0.0, 0, 3, 3, latency_s=0.3),
        TaskResult("k1", "knowledge", "Paris", 1.0, 0, 1, 1, latency_s=0.2),
    ]
    stats = evaluate_by_category(results)
    assert stats["math"].n == 2
    assert stats["math"].accuracy == 0.5
    assert stats["math"].total_tokens == 5
    assert stats["math"].avg_latency_s == 0.2
    assert stats["knowledge"].accuracy == 1.0


def test_assess_local_viability_buckets_correctly():
    fast_correct = [TaskResult("a", "easy", "ok", 1.0, 0, 1, 1, latency_s=1.0)]
    slow_correct = [TaskResult("b", "slow", "ok", 1.0, 0, 1, 1, latency_s=45.0)]
    fast_wrong = [TaskResult("c", "hard", "no", 0.0, 0, 1, 1, latency_s=1.0)]
    viability = assess_local_viability(
        fast_correct + slow_correct + fast_wrong, accuracy_floor=0.80, latency_budget_s=30.0)

    assert viability["easy"].local_viable is True
    assert "fast + likely-correct" in viability["easy"].reason

    assert viability["slow"].local_viable is False
    assert "budget" in viability["slow"].reason

    assert viability["hard"].local_viable is False
    assert "accuracy floor" in viability["hard"].reason


def test_run_local_probe_scores_and_times_each_task():
    client = StubClient(lambda kw: stub_response([kw["messages"][-1]["content"].split("?")[0]
                                                   .replace("2+2", "4").replace("10-3", "7")
                                                   .replace("capital of France", "Paris")]))
    local = LocalRunner(client, model="local-stub")
    results = run_local_probe(local, TASKS)

    assert len(results) == 3
    by_id = {r.task_id: r for r in results}
    assert by_id["m1"].score == 1.0
    assert by_id["m1"].category == "math"
    assert all(r.latency_s is not None and r.latency_s >= 0 for r in results)


def test_run_model_over_tasks_and_rank_models_by_category_prefers_cheap_correct():
    def handler(kw):
        model = kw["model"]
        prompt = kw["messages"][-1]["content"]
        gold_map = {"2+2?": "4", "10-3?": "7", "capital of France?": "Paris"}
        correct = gold_map[prompt]
        if model == "cheap-good":
            return stub_response([correct], usage={"prompt_tokens": 5, "completion_tokens": 1,
                                                    "total_tokens": 6})
        if model == "expensive-good":
            return stub_response([correct], usage={"prompt_tokens": 50, "completion_tokens": 10,
                                                    "total_tokens": 60})
        return stub_response(["nope"], usage={"prompt_tokens": 5, "completion_tokens": 1,
                                              "total_tokens": 6})

    client = StubClient(handler)
    per_model = {m: run_model_over_tasks(client, m, TASKS)
                for m in ("cheap-good", "expensive-good", "bad")}

    ranking = rank_models_by_category(TASKS, per_model, accuracy_floor=0.80)

    math_rank = [r.model for r in ranking["math"]]
    assert math_rank[0] == "cheap-good"          # clears floor, fewest tokens
    assert math_rank[1] == "expensive-good"      # clears floor, more tokens
    assert math_rank[2] == "bad"                 # never clears floor -> ranked last
    assert ranking["math"][0].clears_floor is True
    assert ranking["math"][2].clears_floor is False


def test_run_bakeoff_skips_already_cached_models():
    calls = {"n": 0}

    def handler(kw):
        calls["n"] += 1
        return stub_response(["4"], usage={"prompt_tokens": 1, "completion_tokens": 1,
                                           "total_tokens": 2})

    client = StubClient(handler)
    cached = {"already-run": run_model_over_tasks(StubClient(handler), "already-run", TASKS)}
    calls["n"] = 0  # reset after seeding the cache

    per_model, ranking = run_bakeoff(TASKS, ["already-run", "new-model"], client,
                                     accuracy_floor=0.5, cached=cached)

    assert calls["n"] == len(TASKS)              # only the uncached model made live calls
    assert set(per_model) == {"already-run", "new-model"}
    assert "math" in ranking


def test_run_model_over_tasks_uses_classifier_when_given():
    """Real practice tasks carry no 'category' field at all -- without a
    classifier every task falls into 'uncategorized' and the bake-off can't
    tell categories apart. Calibrated 2026-07-09 (D28): 'sentiment' is a
    literal token, so this is exactly what routing_policy.default.json's
    category-specific templates depend on downstream."""
    from routing_eval.classify import KeywordClassifier

    uncategorized_tasks = [{"task_id": "t1", "prompt": "Classify the sentiment of this comment."}]
    client = StubClient(stub_response(["Positive: great."]))

    no_classifier = run_model_over_tasks(client, "m", uncategorized_tasks)
    assert no_classifier[0].category == "uncategorized"

    with_classifier = run_model_over_tasks(client, "m", uncategorized_tasks,
                                           classifier=KeywordClassifier())
    assert with_classifier[0].category == "sentiment"


def test_run_model_over_tasks_applies_the_matching_category_template():
    from routing_eval.classify import KeywordClassifier

    tasks = [{"task_id": "t1", "prompt": "Classify the sentiment of this comment."}]
    client = StubClient(stub_response(["Positive: great."]))
    category_templates = {"sentiment": "sentiment_with_justification", "_default": "default"}

    run_model_over_tasks(client, "m", tasks, classifier=KeywordClassifier(),
                         category_templates=category_templates)

    system_msg = client.calls[0]["messages"][0]
    assert system_msg["role"] == "system"
    assert "justification" in system_msg["content"]


def test_rank_models_by_category_derives_categories_from_results_not_tasks():
    """Real practice tasks have no category field -- rank_models_by_category
    must not silently collapse everything into one 'uncategorized' bucket
    when the actual TaskResults were classified with real categories."""
    from routing_eval.classify import KeywordClassifier

    bare_tasks = [{"task_id": "t1", "prompt": "Classify the sentiment of this comment."},
                 {"task_id": "t2", "prompt": "Write a python function to add two numbers."}]
    client = StubClient(stub_response(["ok"], usage={"prompt_tokens": 1, "completion_tokens": 1,
                                                      "total_tokens": 2}))
    per_model = {"m1": run_model_over_tasks(client, "m1", bare_tasks,
                                            classifier=KeywordClassifier())}

    ranking = rank_models_by_category(bare_tasks, per_model, accuracy_floor=0.0)

    assert set(ranking) == {"sentiment", "code_gen"}


def test_task_results_round_trip(tmp_path):
    results = [TaskResult("t1", "math", "4", 1.0, 0, 2, 2, latency_s=0.5)]
    path = tmp_path / "results.json"
    save_task_results(results, str(path))
    loaded = load_task_results(str(path))
    assert loaded == results


def test_bakeoff_records_round_trip(tmp_path):
    per_model = {"m1": [TaskResult("t1", "math", "4", 1.0, 1, 1, 2)]}
    path = tmp_path / "bakeoff.json"
    save_bakeoff_records(per_model, str(path))
    loaded = load_bakeoff_records(str(path))
    assert loaded == per_model
    # sanity: it's plain JSON, not something exotic
    with open(path) as f:
        raw = json.load(f)
    assert raw["m1"][0]["task_id"] == "t1"
