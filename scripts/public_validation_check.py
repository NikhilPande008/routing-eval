#!/usr/bin/env python3
"""Run the reconstructed public validation set (scripts/fixtures/
public_validation.json, from the official Judging FAQ v2 shorts) through the
FULL deployed path -- TieredClassifier, checked-in policy, local tier if
LOCAL_BASE_URL is set, real Fireworks otherwise/escalation -- and print each
answer next to its official rubric for hand-checking. Dev-token spend only.

Run:
  set -a && source .env && set +a
  export LOCAL_BASE_URL=http://127.0.0.1:8091/v1 LOCAL_MODEL=qwen2.5-3b-instruct
  python3 scripts/public_validation_check.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval.llm import OpenAICompatibleClient  # noqa: E402
from routing_eval.llm.runners import LocalRunner  # noqa: E402
from routing_eval.modelids import split_models  # noqa: E402
from routing_eval.policy import DEFAULT_POLICY_PATH, PolicyRouter, load_policy  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "public_validation.json")


def main() -> int:
    client = OpenAICompatibleClient(os.environ["FIREWORKS_BASE_URL"],
                                    api_key=os.environ["FIREWORKS_API_KEY"])
    allowed = split_models(os.environ["ALLOWED_MODELS"])
    local = None
    if os.environ.get("LOCAL_BASE_URL"):
        local = LocalRunner(OpenAICompatibleClient(os.environ["LOCAL_BASE_URL"]),
                            model=os.environ.get("LOCAL_MODEL", "local-model"),
                            logprobs=False)
        print(f"local tier ENABLED via {os.environ['LOCAL_BASE_URL']}\n", file=sys.stderr)

    router = PolicyRouter(policy=load_policy(DEFAULT_POLICY_PATH), remote_client=client,
                          allowed_models=allowed, local=local)
    with open(FIXTURE) as f:
        tasks = json.load(f)["tasks"]

    for i, t in enumerate(tasks):
        cls = router.classifier.classify(t["prompt"])
        r = router.route_task({"task_id": t["task_id"], "prompt": t["prompt"]}, i)
        print(f"{'='*100}\n{t['task_id']} [{t['category']}] detected={cls.category}")
        print(f"RUBRIC: {t['rubric']}")
        if "gold" in t:
            print(f"GOLD:   {t['gold']}")
        print(f"ANSWER: {r['answer']}\n")
    router.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
