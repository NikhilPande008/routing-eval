"""Shared task-dict helpers: load /input/tasks.json-shaped files and extract
task_id / prompt text. Used by conformance.py (the scoring shell),
modelselect.py (the Step 2 measurement rig), and policy.py (the Step 3
routing policy) -- all three need to agree on what a "task" looks like, and
none of them should import each other just to get this.

The guide (D20) confirmed task_id and the {task_id, answer} output shape.
Confirmed 2026-07-09 against the real 8 practice tasks
(scripts/fixtures/practice_tasks.json): the prompt field is literally
"prompt" -- already `_PROMPT_FIELDS`'s first-checked entry, so no change was
needed. The fallback chain (question/input/text) and the loud per-task
failure stay as defense-in-depth in case a different task set uses a
different field name.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

_PROMPT_FIELDS = ("prompt", "question", "input", "text")


def load_tasks(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of tasks (or {{'tasks': [...]}})")
    return data


def task_id(task: Dict[str, Any], idx: int) -> str:
    for key in ("task_id", "id"):
        if key in task:
            return str(task[key])
    raise ValueError(f"task #{idx} has no 'task_id' or 'id' field: {task!r}")


def task_prompt(task: Dict[str, Any], idx: int) -> str:
    for key in _PROMPT_FIELDS:
        if key in task:
            return str(task[key])
    raise ValueError(
        f"task #{idx} has none of {_PROMPT_FIELDS} -- update _PROMPT_FIELDS "
        f"once the real tasks.json schema is confirmed: {task!r}")
