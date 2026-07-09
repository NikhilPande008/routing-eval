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
from typing import Optional

from .classify import Classifier
from .llm import OpenAICompatibleClient
from .llm.runners import LocalRunner
from .modelids import split_models
from .policy import (DEFAULT_LOW_CONFIDENCE_THRESHOLD, DEFAULT_POLICY_PATH,
                     DEFAULT_TIMEOUT_S, PolicyRouter, load_policy)
from .taskio import load_tasks

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


def run(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT,
        client: Optional[OpenAICompatibleClient] = None,
        policy_path: Optional[str] = None,
        classifier: Optional[Classifier] = None,
        local: Optional[LocalRunner] = None,
        low_confidence_threshold: Optional[float] = None,
        default_timeout_s: Optional[float] = None) -> int:
    allowed_models = split_models(_require_env("ALLOWED_MODELS"))
    if client is None:
        base_url = _require_env("FIREWORKS_BASE_URL")
        api_key = _require_env("FIREWORKS_API_KEY")
        client = OpenAICompatibleClient(base_url, api_key=api_key)

    if local is None:
        local_base_url = os.environ.get("LOCAL_BASE_URL")
        if local_base_url:
            local = LocalRunner(OpenAICompatibleClient(local_base_url),
                                model=os.environ.get("LOCAL_MODEL", "local-model"))

    if low_confidence_threshold is None:
        low_confidence_threshold = _env_float("ROUTING_LOW_CONFIDENCE_THRESHOLD",
                                              DEFAULT_LOW_CONFIDENCE_THRESHOLD)
    if default_timeout_s is None:
        default_timeout_s = _env_float("ROUTING_TIMEOUT_S", DEFAULT_TIMEOUT_S)

    router = PolicyRouter(
        policy=load_policy(policy_path or DEFAULT_POLICY_PATH),
        remote_client=client, allowed_models=allowed_models, classifier=classifier,
        local=local, low_confidence_threshold=low_confidence_threshold,
        default_timeout_s=default_timeout_s)

    tasks = load_tasks(input_path)
    results = router.route_all(tasks)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"conformance: wrote {len(results)} results -> {output_path}")
    return 0
