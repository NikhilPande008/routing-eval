"""Scoring-VM conformance shell: /input/tasks.json -> /output/results.json.

Contract per the official Track 1 participant guide (2026-07-08, see
DECISIONS.md D17-D20): the grading VM mounts /input/tasks.json, runs the
container, and reads /output/results.json once it exits 0. Env injected by
the harness: FIREWORKS_BASE_URL, FIREWORKS_API_KEY, ALLOWED_MODELS
(comma/whitespace-separated).

Step 1 proved the I/O + call contract alone (one task, one Fireworks call).
Step 3 (routing_eval.policy) now decides HOW each task is answered --
`run()` delegates to a PolicyRouter built from the checked-in safe-default
policy (routes everything to Fireworks, first ALLOWED_MODELS model --
IDENTICAL to Step 1's behavior) unless a calibrated routing_policy.json or an
optional local model (LOCAL_BASE_URL / LOCAL_MODEL env) is wired in.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

from .classify import Classifier
from .llm import OpenAICompatibleClient
from .llm.runners import LocalRunner
from .modelids import split_models
from .policy import (DEFAULT_LOW_CONFIDENCE_THRESHOLD, DEFAULT_POLICY_PATH,
                     DEFAULT_TIMEOUT_S, PolicyRouter, load_policy)
from .taskio import load_tasks, task_id

DEFAULT_INPUT = "/input/tasks.json"
DEFAULT_OUTPUT = "/output/results.json"


def _model(allowed_models: str) -> str:
    return split_models(allowed_models)[0]


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"{name} is not set -- required by the scoring contract")
    return val


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default


def _write_results(output_path: str, results) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def _empty_results(tasks) -> list:
    results = []
    for idx, task in enumerate(tasks):
        try:
            tid = task_id(task, idx)
        except ValueError:
            tid = f"unknown-{idx}"
        results.append({"task_id": tid, "answer": ""})
    return results


def run(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT,
        client: Optional[OpenAICompatibleClient] = None,
        policy_path: Optional[str] = None,
        classifier: Optional[Classifier] = None,
        local: Optional[LocalRunner] = None,
        low_confidence_threshold: Optional[float] = None,
        default_timeout_s: Optional[float] = None) -> int:
    try:
        tasks = load_tasks(input_path)
    except FileNotFoundError:
        print(f"conformance: {input_path} not found -- writing empty results", file=sys.stderr)
        _write_results(output_path, [])
        return 0

    try:
        allowed_models = split_models(_require_env("ALLOWED_MODELS"))
    except Exception as e:  # noqa: BLE001 -- malformed/missing env should not crash scoring
        print(f"conformance: Fireworks env unavailable ({e}) -- writing empty answers",
              file=sys.stderr)
        _write_results(output_path, _empty_results(tasks))
        return 0

    if client is None:
        try:
            base_url = _require_env("FIREWORKS_BASE_URL")
            api_key = _require_env("FIREWORKS_API_KEY")
        except Exception as e:  # noqa: BLE001 -- keep the container alive for evaluation
            print(f"conformance: Fireworks env unavailable ({e}) -- writing empty answers",
                  file=sys.stderr)
            _write_results(output_path, _empty_results(tasks))
            return 0
        client = OpenAICompatibleClient(base_url, api_key=api_key)

    if local is None:
        local_base_url = os.environ.get("LOCAL_BASE_URL")
        if local_base_url:
            # logprobs=False: the score path never reads them, and not asking
            # keeps the llama-server request minimal (gates, which do need
            # logprobs, build their own LocalRunner). LOCAL_MAX_TOKENS (D48
            # redesign): caps local generations so a runaway answer can't eat
            # the per-request timeout; the entrypoint exports 128.
            local = LocalRunner(OpenAICompatibleClient(local_base_url),
                                model=os.environ.get("LOCAL_MODEL", "local-model"),
                                max_tokens=int(os.environ.get("LOCAL_MAX_TOKENS", "512")),
                                logprobs=False)

    if low_confidence_threshold is None:
        low_confidence_threshold = _env_float("ROUTING_LOW_CONFIDENCE_THRESHOLD",
                                              DEFAULT_LOW_CONFIDENCE_THRESHOLD)
    if default_timeout_s is None:
        default_timeout_s = _env_float("ROUTING_TIMEOUT_S", DEFAULT_TIMEOUT_S)

    # D52/D54: the batch-level local time budget (PolicyRouter.local_budget_s).
    # 330s: tightened from 380 after D54's retry loop pushed the emulated
    # 19-task worst case to 406s wall -- 330 + model load + the full remote
    # sweep stays under ~520s of the 600s total (D19) even worst-case.
    local_budget_s = _env_float("ROUTING_LOCAL_BUDGET_S", 330.0) if local else None

    # POLICY_PATH env selects the policy file so ONE image serves both
    # submission lines (2026-07-12): unset/empty => the checked-in kimi
    # default (DEFAULT_POLICY_PATH), UNCHANGED; the Gemma line's image bakes
    # POLICY_PATH=/app/routing_eval/routing_policy.gemma.json via a Docker
    # build arg. An explicit policy_path arg (tests/CLI) still wins over env.
    resolved_policy_path = (policy_path or (os.environ.get("POLICY_PATH") or "").strip()
                            or DEFAULT_POLICY_PATH)
    if resolved_policy_path != DEFAULT_POLICY_PATH:
        print(f"conformance: using policy {resolved_policy_path}", file=sys.stderr)

    router = PolicyRouter(
        policy=load_policy(resolved_policy_path),
        remote_client=client, allowed_models=allowed_models, classifier=classifier,
        local=local, low_confidence_threshold=low_confidence_threshold,
        default_timeout_s=default_timeout_s, local_budget_s=local_budget_s)

    results = router.route_all(tasks)

    _write_results(output_path, results)
    print(f"conformance: wrote {len(results)} results -> {output_path}")
    return 0
