#!/usr/bin/env python3
"""Live smoke test against real endpoints. Skips cleanly unless credentials are
set, so routing:verify can call it in any environment.

Env:
  FIREWORKS_API_KEY   required for the remote check
  FIREWORKS_MODEL     default: accounts/fireworks/models/llama-v3p1-8b-instruct
  VLLM_BASE_URL       optional; if set, also smoke-tests the local runner
  VLLM_MODEL          default: the model name your vLLM server serves

Run:  python -m scripts.live_smoke      (or  python scripts/live_smoke.py )
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing_eval.llm import LocalRunner, OpenAICompatibleClient, RemoteRunner  # noqa: E402
from routing_eval.schema import Item  # noqa: E402

ITEM = Item("smoke", "math", "What is 17 times 4?", 68, "numeric", {})


def main() -> int:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        print("SKIP: FIREWORKS_API_KEY not set (no live call made).")
        return 0

    fw_model = os.environ.get("FIREWORKS_MODEL",
                              "accounts/fireworks/models/llama-v3p1-8b-instruct")
    remote = RemoteRunner(
        OpenAICompatibleClient("https://api.fireworks.ai/inference/v1", api_key=key),
        model=fw_model, max_tokens=16, stop=["\n"])
    out = remote.run(ITEM)
    print(f"[remote] model={fw_model}")
    print(f"  answer={out.answer!r}  tokens: prompt={out.prompt_tokens} "
          f"completion={out.completion_tokens} total={out.total_tokens}")

    vllm_url = os.environ.get("VLLM_BASE_URL")
    if vllm_url:
        vllm_model = os.environ.get("VLLM_MODEL", "local-model")
        local = LocalRunner(OpenAICompatibleClient(vllm_url), model=vllm_model,
                            max_tokens=16, n_samples=3, temperature=0.7)
        lo = local.run(ITEM)
        print(f"[local]  model={vllm_model}")
        print(f"  answer={lo.answer!r}  samples={lo.samples}  "
              f"logprob_tokens={len(lo.token_logprobs)}")
    else:
        print("[local]  SKIP: VLLM_BASE_URL not set.")
    print("live smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
