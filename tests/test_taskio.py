import json

import pytest

from routing_eval.taskio import load_tasks, task_id, task_prompt


def test_task_id_reads_task_id_or_id():
    assert task_id({"task_id": "t1"}, 0) == "t1"
    assert task_id({"id": "t2"}, 0) == "t2"
    with pytest.raises(ValueError):
        task_id({}, 0)


def test_task_prompt_fallback_chain():
    assert task_prompt({"prompt": "p"}, 0) == "p"
    assert task_prompt({"question": "q"}, 0) == "q"
    assert task_prompt({"input": "i"}, 0) == "i"
    assert task_prompt({"text": "x"}, 0) == "x"
    with pytest.raises(ValueError):
        task_prompt({"nope": "n"}, 0)


def test_load_tasks_accepts_a_bare_list(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"task_id": "t1", "prompt": "hi"}]))
    assert load_tasks(str(path)) == [{"task_id": "t1", "prompt": "hi"}]


def test_load_tasks_accepts_a_tasks_wrapped_dict(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [{"task_id": "t1", "prompt": "hi"}]}))
    assert load_tasks(str(path)) == [{"task_id": "t1", "prompt": "hi"}]


def test_load_tasks_rejects_the_wrong_shape(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"not_tasks": []}))
    with pytest.raises(ValueError):
        load_tasks(str(path))
