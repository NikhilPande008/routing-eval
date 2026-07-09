import json

import pytest

from routing_eval.conformance import _model, run
from routing_eval.llm import StubClient, stub_response


def test_model_picks_first_of_comma_or_space_separated():
    assert _model("modelA,modelB") == "accounts/fireworks/models/modelA"
    assert _model(" modelA , modelB ") == "accounts/fireworks/models/modelA"
    assert _model("modelA modelB") == "accounts/fireworks/models/modelA"
    with pytest.raises(ValueError):
        _model("   ")


def test_run_writes_valid_results_shape(tmp_path, monkeypatch):
    tasks = [{"task_id": "t1", "prompt": "2+2?"}, {"id": "t2", "question": "capital of France?"}]
    input_path = tmp_path / "tasks.json"
    input_path.write_text(json.dumps(tasks))
    output_path = tmp_path / "results.json"

    # ALLOWED_MODELS is still required (and still what a null-model category
    # would fall back to), but the checked-in default policy pins every
    # category to kimi-k2p7-code (D30) -- irrelevant here since the stub
    # answers regardless of which model string it's called with.
    monkeypatch.setenv("ALLOWED_MODELS", "stub-model")
    stub = StubClient(stub_response(["stub-answer"]))

    rc = run(str(input_path), str(output_path), client=stub)

    assert rc == 0
    results = json.loads(output_path.read_text())
    assert results == [
        {"task_id": "t1", "answer": "stub-answer"},
        {"task_id": "t2", "answer": "stub-answer"},
    ]
    assert stub.calls[0]["model"] == "accounts/fireworks/models/kimi-k2p7-code"


def test_run_survives_a_malformed_task(tmp_path, monkeypatch):
    tasks = [{"task_id": "good", "prompt": "hi"}, {"task_id": "no-prompt-field"}]
    input_path = tmp_path / "tasks.json"
    input_path.write_text(json.dumps(tasks))
    output_path = tmp_path / "results.json"

    monkeypatch.setenv("ALLOWED_MODELS", "stub-model")
    stub = StubClient(stub_response(["ok"]))

    rc = run(str(input_path), str(output_path), client=stub)

    assert rc == 0
    results = json.loads(output_path.read_text())
    assert results == [
        {"task_id": "good", "answer": "ok"},
        {"task_id": "no-prompt-field", "answer": ""},
    ]


def test_run_requires_allowed_models_env(tmp_path, monkeypatch):
    input_path = tmp_path / "tasks.json"
    input_path.write_text("[]")
    monkeypatch.delenv("ALLOWED_MODELS", raising=False)
    with pytest.raises(RuntimeError):
        run(str(input_path), str(tmp_path / "results.json"), client=StubClient(stub_response(["x"])))


def test_run_creates_output_dir(tmp_path, monkeypatch):
    input_path = tmp_path / "tasks.json"
    input_path.write_text(json.dumps([{"task_id": "t1", "prompt": "hi"}]))
    output_path = tmp_path / "nested" / "dir" / "results.json"

    monkeypatch.setenv("ALLOWED_MODELS", "m")
    rc = run(str(input_path), str(output_path), client=StubClient(stub_response(["ok"])))

    assert rc == 0
    assert output_path.exists()
