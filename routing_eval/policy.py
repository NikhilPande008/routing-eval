"""Step 3: the per-category routing policy that turns the conformance shell
(Step 1) into the actual agent, calibrated by Step 2 (probe-local + bakeoff).

routing_policy.json shape:
  {"<category>": {"tier": "local" | "fireworks",
                  "model": "<fireworks-model-id>" | null,
                  "max_tokens": <int>,
                  "prompt_template": "<name>",
                  "timeout_s": <float, optional>},
   "_default": {...same shape, used for any category not listed}}

`model: null` on a "fireworks" entry means "use whatever ALLOWED_MODELS gives
at runtime" -- the same behavior the conformance shell had before this policy
existed. No model ID or threshold in this file is hardcoded in Python; the
one numeric default in code (DEFAULT_TIMEOUT_S) is D19's own published
per-request budget, not a tuned value, and is still overridable via the
ROUTING_TIMEOUT_S env var or the `timeout_s` CLI/API param.

DEFAULT_POLICY_PATH ships a checked-in SAFE DEFAULT (routing_policy.default.json,
package data): every category falls through "_default" -> tier "fireworks",
model null. That is IDENTICAL to the conformance shell's pre-policy behavior
(Step 1) -- routing through this module changes nothing observable until a
calibrated policy is generated and swapped in.

generate_policy() is what upgrades that default to real per-category
tiers/models, fed by Step 2's probe-local (local-viability) and bakeoff
(per-category model ranking) outputs. Run it AFTER launch-day calibration;
its output is a DRAFT (routing_policy.draft.json by default) -- review before
replacing the checked-in default.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .classify import Classifier, KeywordClassifier
from .llm.runners import LocalRunner, RemoteRunner
from .modelids import normalize_model_id
from .modelselect import LocalViability, ModelCategoryRanking
from .prompts import CODE_ONLY_SYSTEM, PROMPT_TEMPLATES, SENTIMENT_SYSTEM  # noqa: F401 (re-exported)
from .prompts import system_prompt as _system_prompt
from .schema import Item
from .taskio import task_id as _task_id
from .taskio import task_prompt as _task_prompt

DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(__file__), "routing_policy.default.json")
DEFAULT_TIMEOUT_S = 30.0          # D19: the grading VM's own per-request budget, not a tuned value
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.5   # arbitrary until real calibration data exists; override freely
_DEFAULT_KEY = "_default"

# CODE_ONLY_SYSTEM, SENTIMENT_SYSTEM, PROMPT_TEMPLATES now live in
# routing_eval.prompts (shared with modelselect.py's bake-off; see that
# module's docstring for why). Re-exported here for backward compatibility --
# existing callers of `from routing_eval.policy import PROMPT_TEMPLATES` etc.
# keep working unchanged.

# Backstop for "code_only": the instruction alone doesn't guarantee
# compliance (models fence code out of habit even when told not to).
# Deterministic and harmless when there's no fence to strip.
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\n(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(answer: str) -> str:
    m = _CODE_FENCE_RE.match(answer.strip())
    return m.group(1) if m else answer


def _is_blank(answer: str) -> bool:
    return not answer or not answer.strip()


@dataclass
class PolicyEntry:
    tier: str                              # "local" | "fireworks"
    model: Optional[str] = None            # None on "fireworks" => first ALLOWED_MODELS at runtime
    max_tokens: int = 256
    prompt_template: str = "default"
    timeout_s: Optional[float] = None      # None => caller's default_timeout_s (D19 budget)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyEntry":
        return cls(tier=d["tier"], model=d.get("model"), max_tokens=int(d.get("max_tokens", 256)),
                   prompt_template=d.get("prompt_template", "default"),
                   timeout_s=d.get("timeout_s"))


SAFE_FALLBACK_ENTRY = PolicyEntry(tier="fireworks", model=None, max_tokens=256,
                                  prompt_template="default", timeout_s=None)


def load_policy(path: str = DEFAULT_POLICY_PATH) -> Dict[str, PolicyEntry]:
    with open(path) as f:
        raw = json.load(f)
    return {cat: PolicyEntry.from_dict(entry) for cat, entry in raw.items()}


def resolve_entry(policy: Dict[str, PolicyEntry], category: str) -> PolicyEntry:
    """Category not in the policy -> "_default" -> the safe in-code fallback
    (only reachable if the policy FILE itself is missing "_default", which
    the checked-in default never is)."""
    return policy.get(category) or policy.get(_DEFAULT_KEY) or SAFE_FALLBACK_ENTRY


def _resolve_model(entry: PolicyEntry, allowed_models: List[str]) -> str:
    if entry.model:
        return normalize_model_id(entry.model)   # hand-authored entries may be bare names
    if not allowed_models:
        raise ValueError("policy entry has no model and ALLOWED_MODELS is empty")
    return allowed_models[0]   # already normalized -- came from split_models()


class PolicyRouter:
    """The score-path decision maker: task -> classify category -> policy
    entry -> local-or-fireworks call -> {task_id, answer}.

    Owns a small thread pool used only to enforce the per-request timeout on
    local calls (a StubClient / any client that doesn't do real socket I/O
    can't honor a `timeout` kwarg itself, so this bounds the wait
    independently of the client). Call close() once per batch -- shutdown is
    non-blocking (wait=False): if a local call is genuinely wedged with no
    timeout support of its own, this router still emits the Fireworks
    fallback answer and moves on; it does not guarantee killing the stray
    thread (Python threads cannot be force-killed), but the client-level
    `timeout` passed to LocalRunner.run() is what actually terminates a real
    HTTP call at the socket level.
    """

    def __init__(self, policy: Dict[str, PolicyEntry], remote_client,
                allowed_models: List[str], classifier: Optional[Classifier] = None,
                local: Optional[LocalRunner] = None,
                low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
                default_timeout_s: float = DEFAULT_TIMEOUT_S):
        self.policy = policy
        self.remote_client = remote_client
        self.allowed_models = allowed_models
        self.classifier = classifier or KeywordClassifier()
        self.local = local
        self.low_confidence_threshold = low_confidence_threshold
        self.default_timeout_s = default_timeout_s
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2) if local else None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def _pick_retry_model(self, used_model: str) -> Optional[str]:
        """A different model from ALLOWED_MODELS to retry with -- never a
        hardcoded model ID; whichever models are actually injected at
        runtime decide what "the other model" is. Picks the first entry
        that isn't the one just tried; None if ALLOWED_MODELS has no
        alternative (e.g. only one model configured)."""
        for m in self.allowed_models:
            if m != used_model:
                return m
        return None

    def _try_fireworks(self, entry: PolicyEntry, item: Item, model: str) -> str:
        """One attempt against a specific model. Never raises -- any failure
        (exception or a blank response) returns "" so the caller can retry
        with a different model uniformly regardless of cause. `timeout` is
        D19's per-request budget (30s default), so two attempts (primary +
        retry) still fit well inside the 10-minute total even in the worst
        case where every task needs a retry."""
        system = _system_prompt(entry.prompt_template)
        timeout_s = entry.timeout_s or self.default_timeout_s
        try:
            remote = RemoteRunner(self.remote_client, model=model, max_tokens=entry.max_tokens,
                                  system=system)
            out = remote.run(item, timeout=timeout_s)
            print(f"policy: task {item.id} answered by {model} -- "
                 f"prompt={out.prompt_tokens} completion={out.completion_tokens} "
                 f"total={out.total_tokens} tokens", file=sys.stderr)
            answer = out.answer
            if entry.prompt_template == "code_only":
                answer = _strip_code_fence(answer)
            return answer
        except Exception as e:  # noqa: BLE001 -- caller decides retry-vs-empty uniformly
            print(f"policy: task {item.id} call to {model} failed -- {e}", file=sys.stderr)
            return ""

    def route_task(self, task: Dict[str, Any], idx: int) -> Dict[str, Any]:
        try:
            task_id = _task_id(task, idx)
        except ValueError as e:
            print(f"policy: {e}", file=sys.stderr)
            return {"task_id": f"unknown-{idx}", "answer": ""}

        try:
            prompt = _task_prompt(task, idx)
        except ValueError as e:
            print(f"policy: task {task_id} failed -- {e}", file=sys.stderr)
            return {"task_id": task_id, "answer": ""}

        cls = self.classifier.classify(prompt)
        entry = resolve_entry(self.policy, cls.category)
        flag = " LOW-CONFIDENCE" if cls.confidence < self.low_confidence_threshold else ""
        print(f"policy: task {task_id} category={cls.category!r} "
             f"confidence={cls.confidence:.2f}{flag} tier={entry.tier}", file=sys.stderr)

        item = Item(id=task_id, task_type=cls.category, input=prompt, gold=None, scorer="exact")

        if entry.tier == "local" and self.local is not None:
            timeout_s = entry.timeout_s or self.default_timeout_s
            try:
                future = self._executor.submit(self.local.run, item, timeout_s)
                out = future.result(timeout=timeout_s)
                return {"task_id": task_id, "answer": out.answer}
            except Exception as e:  # noqa: BLE001 -- timeout or any local failure degrades to Fireworks
                print(f"policy: local call for {task_id} timed out/failed ({e}) "
                     f"-- falling back to Fireworks", file=sys.stderr)

        answer = ""
        try:
            primary_model = _resolve_model(entry, self.allowed_models)
            answer = self._try_fireworks(entry, item, primary_model)
            if _is_blank(answer):
                retry_model = self._pick_retry_model(primary_model)
                if retry_model is not None:
                    print(f"policy: task {task_id} blank/failed answer from {primary_model} "
                         f"-- retrying once with {retry_model} (empty is the last resort, "
                         f"not the first fallback)", file=sys.stderr)
                    answer = self._try_fireworks(entry, item, retry_model)
                    if _is_blank(answer):
                        print(f"policy: task {task_id} still blank after retry with "
                             f"{retry_model} -- falling back to empty answer", file=sys.stderr)
                else:
                    print(f"policy: task {task_id} blank/failed and no alternate model in "
                         f"ALLOWED_MODELS to retry with -- empty answer", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 -- never crash the batch over one task
            print(f"policy: task {task_id} routing failed unexpectedly -- {e}", file=sys.stderr)
            answer = ""
        return {"task_id": task_id, "answer": answer}

    def route_all(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            return [self.route_task(t, i) for i, t in enumerate(tasks)]
        finally:
            self.close()


# ---------------------------------------------------------------------------
# Draft policy generator, fed by Step 2's probe-local + bakeoff outputs.
# ---------------------------------------------------------------------------

def generate_policy(local_viability: Dict[str, LocalViability],
                    bakeoff_ranking: Dict[str, List[ModelCategoryRanking]],
                    max_tokens: int = 256, prompt_template: str = "default",
                    ) -> Dict[str, Dict[str, Any]]:
    categories = sorted(set(local_viability) | set(bakeoff_ranking))
    policy: Dict[str, Dict[str, Any]] = {}
    for cat in categories:
        viability = local_viability.get(cat)
        if viability is not None and viability.local_viable:
            policy[cat] = {"tier": "local", "model": None, "max_tokens": max_tokens,
                           "prompt_template": prompt_template}
            continue

        rows = bakeoff_ranking.get(cat, [])
        clearing = [r for r in rows if r.clears_floor]
        if clearing:
            model = clearing[0].model     # rank_models_by_category already sorts cheapest-first
        else:
            model = None                  # nothing clears the floor -- don't pretend to pick
            if rows:
                print(f"policy-gen: no model clears the accuracy floor for category "
                     f"{cat!r} -- leaving model unset (falls back to first ALLOWED_MODELS "
                     f"at runtime)", file=sys.stderr)
        policy[cat] = {"tier": "fireworks", "model": model, "max_tokens": max_tokens,
                       "prompt_template": prompt_template}

    policy[_DEFAULT_KEY] = {"tier": "fireworks", "model": None, "max_tokens": max_tokens,
                            "prompt_template": prompt_template}
    return policy


def save_policy(policy: Dict[str, Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(policy, f, indent=2)
