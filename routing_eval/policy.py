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

2026-07-09 (D37): PolicyRouter's default classifier is now TwoWayClassifier
(classify.py), not the 8-way KeywordClassifier. D35 found kimi-k2p7-code
comparable-or-better than minimax-m3 in every category, so an 8-way split no
longer changes which model gets called -- the only thing category still
controls is which of 3 prompt templates is used (code_only /
sentiment_with_justification / generic default), and D29/D36 showed the
8-way classifier doesn't reliably get even THAT right. TwoWayClassifier
detects only "code" and "sentiment"; everything else is "general", which
isn't a key in routing_policy.default.json so it falls through "_default"
exactly like an unmatched 8-way category always did. The checked-in default
below keeps "code_debug"/"code_gen" as separate entries (identical to
"code") purely so KeywordClassifier-based tooling (bakeoff/generate-policy,
still calibrated against the 8-way taxonomy) keeps resolving correctly;
nothing currently classifies into those two names in the deployed path.

2026-07-10 (D41, gate cleared 89.5% -- objective is now tokens, under a
one-task accuracy margin): PolicyRouter's default classifier is now
TieredClassifier (classify.py) -- D37's code/sentiment detectors plus three
new high-precision detectors (entity_extraction / math / logic) that route
to minimal templates. The failure asymmetry that makes this safe: a missed
detection falls through to "_default" (fuller template -- costs tokens,
never accuracy); the detectors are tuned for zero false positives so no
knowledge/summarization task is quietly terse-ified. Every template trim was
judge-proxy-gated before shipping (D8).

2026-07-10 (D40): two independent truncation-eliminating changes, both
accuracy-only. (1) Every `max_tokens` here (PolicyEntry's default, the
checked-in routing_policy.default.json, generate_policy()'s default) moved
256 -> 512 -- a 256-token cap can still truncate a 2-5 sentence explanation
plus a reasoning trace, the exact shape the new "default" template
(prompts.ACCURATE_GENERIC_SYSTEM) now asks for. (2) `_try_fireworks` now
reads and logs `finish_reason` on every call and, if a call comes back
`finish_reason="length"` (truncated), retries ONCE, same model, with a
doubled cap -- a truncated answer is a guaranteed judge failure (D18), so
catching it is worth one extra call. This is orthogonal to and composes
with D31's existing retry-on-blank-with-a-different-model: a length-retry
happens first, inside one model's attempts; if THAT still comes back blank
or truncated-again, the outer blank-check in route_task still swaps models
as before.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .classify import Classifier, TieredClassifier
from .llm.runners import LocalRunner, RemoteRunner
from .localcheck import agreement_problem, local_answer_problem
from .modelids import normalize_model_id
from .modelselect import LocalViability, ModelCategoryRanking
from .prompts import CODE_ONLY_SYSTEM, DIVERSITY_PROMPT_TEMPLATES  # noqa: F401 (re-exported)
from .prompts import PROMPT_TEMPLATES, SENTIMENT_SYSTEM  # noqa: F401 (re-exported)
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
class FireworksAttempt:
    """One `_try_fireworks` call's result -- `answer` is what route_task's
    external `{task_id, answer}` contract needs; `finish_reason`/`total_tokens`
    are for logging and diagnostics (D40), not part of that contract."""
    answer: str
    finish_reason: Optional[str] = None
    total_tokens: int = 0


@dataclass
class PolicyEntry:
    tier: str                              # "local" | "fireworks"
    model: Optional[str] = None            # None on "fireworks" => first ALLOWED_MODELS at runtime
    max_tokens: int = 512                  # D40: 256 truncated multi-sentence answers; 512 is generous
    prompt_template: str = "default"
    timeout_s: Optional[float] = None      # None => caller's default_timeout_s (D19 budget)
    # 2026-07-11 (local tier): template for the LOCAL call only, when it should
    # differ from the remote one. None => prompt_template for both. This keeps
    # the Fireworks ESCALATION path byte-identical to the live-validated remote
    # treatment while the local call gets a prompt written for the small model.
    local_prompt_template: Optional[str] = None
    # 2026-07-12 (D52): 2 => draw a second local sample (temperature 0.7) and
    # require agreement (localcheck.agreement_problem) before keeping -- the
    # content fence for categories whose validators can't check facts or
    # arithmetic (knowledge, math). 1 => single sample, validator-only.
    local_n_samples: int = 1

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyEntry":
        return cls(tier=d["tier"], model=d.get("model"), max_tokens=int(d.get("max_tokens", 512)),
                   prompt_template=d.get("prompt_template", "default"),
                   timeout_s=d.get("timeout_s"),
                   local_prompt_template=d.get("local_prompt_template"),
                   local_n_samples=int(d.get("local_n_samples", 1)))


SAFE_FALLBACK_ENTRY = PolicyEntry(tier="fireworks", model=None, max_tokens=512,
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
                default_timeout_s: float = DEFAULT_TIMEOUT_S,
                local_budget_s: Optional[float] = None):
        self.policy = policy
        self.remote_client = remote_client
        self.allowed_models = allowed_models
        self.classifier = classifier or TieredClassifier()
        self.local = local
        self.low_confidence_threshold = low_confidence_threshold
        self.default_timeout_s = default_timeout_s
        # D52 global time governor: total wall-clock all local attempts may
        # consume across the batch. Replaces reliance on tight per-task
        # timeouts (which slow grading hardware turned into always-escalate):
        # per-task caps can now be generous because the BATCH-level spend is
        # bounded here -- once the budget is gone, every remaining local-tier
        # task goes straight to Fireworks (the proven remote path). None =>
        # unlimited (test/back-compat behavior).
        self.local_budget_s = local_budget_s
        self._local_spent_s = 0.0
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

    def _try_fireworks(self, entry: PolicyEntry, item: Item, model: str,
                       max_tokens: Optional[int] = None) -> FireworksAttempt:
        """One attempt against a specific model. Never raises -- any failure
        (exception or a blank response) returns an empty FireworksAttempt so
        the caller can retry with a different model uniformly regardless of
        cause. `timeout` is D19's per-request budget (30s default), so two
        attempts (primary + retry) still fit well inside the 10-minute total
        even in the worst case where every task needs a retry.

        D40: `finish_reason` is logged on every call. `max_tokens=None` means
        "first attempt at this model" -- if that attempt comes back
        `finish_reason="length"` (truncated), this retries ONCE, same model,
        with a doubled cap (`max_tokens` passed explicitly on the recursive
        call, which is what stops it from ever doubling more than once). A
        truncated answer is a guaranteed judge failure (D18), so this is
        cheap insurance -- at most one extra call, only when actually needed."""
        system = _system_prompt(entry.prompt_template)
        timeout_s = entry.timeout_s or self.default_timeout_s
        tokens_cap = max_tokens or entry.max_tokens
        try:
            remote = RemoteRunner(self.remote_client, model=model, max_tokens=tokens_cap,
                                  system=system)
            out = remote.run(item, timeout=timeout_s)
            print(f"policy: task {item.id} answered by {model} -- "
                 f"prompt={out.prompt_tokens} completion={out.completion_tokens} "
                 f"total={out.total_tokens} tokens finish_reason={out.finish_reason!r} "
                 f"max_tokens={tokens_cap}", file=sys.stderr)
            if out.finish_reason == "length" and max_tokens is None:
                doubled = tokens_cap * 2
                print(f"policy: task {item.id} answer TRUNCATED at max_tokens={tokens_cap} "
                     f"-- retrying once with max_tokens={doubled}", file=sys.stderr)
                return self._try_fireworks(entry, item, model, max_tokens=doubled)
            answer = out.answer
            if entry.prompt_template == "code_only":
                answer = _strip_code_fence(answer)
            return FireworksAttempt(answer, out.finish_reason, out.total_tokens)
        except Exception as e:  # noqa: BLE001 -- caller decides retry-vs-empty uniformly
            print(f"policy: task {item.id} call to {model} failed -- {e}", file=sys.stderr)
            return FireworksAttempt("", None, 0)

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
            if (self.local_budget_s is not None
                    and self._local_spent_s >= self.local_budget_s):
                print(f"policy: task {task_id} local budget exhausted "
                     f"({self._local_spent_s:.0f}s >= {self.local_budget_s:.0f}s) "
                     f"-- going straight to Fireworks", file=sys.stderr)
            else:
                timeout_s = entry.timeout_s or self.default_timeout_s
                local_system = _system_prompt(entry.local_prompt_template
                                              or entry.prompt_template)
                started = time.monotonic()
                try:
                    future = self._executor.submit(self.local.run, item, timeout_s,
                                                   local_system)
                    out = future.result(timeout=timeout_s)
                    # A local answer is only kept if it passes the deterministic
                    # shape check for its category (localcheck.py). Rejection
                    # just means the task pays the remote price it would have
                    # paid anyway -- acceptance of a bad answer is the only
                    # failure mode that costs accuracy, so the check is strict.
                    problem = local_answer_problem(cls.category, prompt, out.answer)
                    # D52/D53 self-consistency: content-unverifiable categories
                    # draw a SECOND sample; the two must agree or the task
                    # escalates. D53: the second sample uses an ALTERNATE
                    # PROMPT WORDING at temperature 0 when one is defined
                    # (DIVERSITY_PROMPT_TEMPLATES) -- prompt diversity
                    # decorrelates arithmetic/deduction slips without the
                    # sampling noise that temp 0.7 injected (observed making
                    # the second math sample WORSE, not just different).
                    # Fallback for templates without an alternate: same
                    # prompt at temperature 0.7 (the D52 behavior). The
                    # first (temperature-0, primary-prompt) sample is the
                    # one delivered.
                    if problem is None and entry.local_n_samples >= 2:
                        alt = DIVERSITY_PROMPT_TEMPLATES.get(
                            entry.local_prompt_template or "")
                        sys2, temp2 = (alt, 0.0) if alt else (local_system, 0.7)
                        future2 = self._executor.submit(self.local.run, item, timeout_s,
                                                        sys2, temp2)
                        out2 = future2.result(timeout=timeout_s)
                        problem = agreement_problem(cls.category, out.answer, out2.answer)
    # D54: one FREE local format-retry before paying for an
                    # escalation. The failure reason is injected into the
                    # retry prompt ("rejected because: X -- fix exactly
                    # that"); the retry must pass the SAME validator and,
                    # for n>=2 categories, a fresh agreement check -- no
                    # fence is weakened, rejections just get a second free
                    # attempt. Skipped when the local budget is gone, and
                    # for PROMPT-property failures no retry can ever fix
                    # (the 19-task timing gate caught retries burning ~15s
                    # each on example-less code tasks -- pure waste).
                    unfixable = problem is not None and problem.startswith(
                        "no verifiable input/output examples")
                    if (problem is not None and not unfixable
                            and (self.local_budget_s is None
                                 or time.monotonic() - started + self._local_spent_s
                                 < self.local_budget_s)):
                        retry_system = (
                            f"{local_system}\nIMPORTANT: your previous attempt "
                            f"was rejected because: {problem}. Produce a "
                            f"corrected answer that fixes exactly that.")
                        print(f"policy: task {task_id} local answer rejected "
                             f"({problem}) -- one free local retry", file=sys.stderr)
                        future_r = self._executor.submit(self.local.run, item,
                                                         timeout_s, retry_system)
                        out_r = future_r.result(timeout=timeout_s)
                        problem_r = local_answer_problem(cls.category, prompt,
                                                         out_r.answer)
                        if problem_r is None and entry.local_n_samples >= 2:
                            alt = DIVERSITY_PROMPT_TEMPLATES.get(
                                entry.local_prompt_template or "")
                            sys2, temp2 = (alt, 0.0) if alt else (local_system, 0.7)
                            future2r = self._executor.submit(self.local.run, item,
                                                             timeout_s, sys2, temp2)
                            out2r = future2r.result(timeout=timeout_s)
                            problem_r = agreement_problem(cls.category, out_r.answer,
                                                          out2r.answer)
                        if problem_r is None:
                            out, problem = out_r, None
                        else:
                            problem = f"{problem}; retry also failed ({problem_r})"
                    if problem is None:
                        self._local_spent_s += time.monotonic() - started
                        answer = out.answer
                        # Local code answers need the same fence-strip the
                        # remote path applies -- fenced code is the D27
                        # judge-fail mode, and gating caught a kept local
                        # answer shipping with ```python fences intact.
                        if entry.prompt_template == "code_only":
                            answer = _strip_code_fence(answer)
                        print(f"policy: task {task_id} answered LOCALLY -- 0 remote tokens "
                             f"({out.tokens} free local completion tokens, "
                             f"local budget used {self._local_spent_s:.0f}s)", file=sys.stderr)
                        return {"task_id": task_id, "answer": answer}
                    print(f"policy: task {task_id} local answer REJECTED ({problem}) "
                         f"-- escalating to Fireworks", file=sys.stderr)
                except Exception as e:  # noqa: BLE001 -- timeout or any local failure degrades to Fireworks
                    print(f"policy: local call for {task_id} timed out/failed ({e}) "
                         f"-- falling back to Fireworks", file=sys.stderr)
                self._local_spent_s += time.monotonic() - started

        answer = ""
        try:
            primary_model = _resolve_model(entry, self.allowed_models)
            answer = self._try_fireworks(entry, item, primary_model).answer
            if _is_blank(answer):
                retry_model = self._pick_retry_model(primary_model)
                if retry_model is not None:
                    print(f"policy: task {task_id} blank/failed answer from {primary_model} "
                         f"-- retrying once with {retry_model} (empty is the last resort, "
                         f"not the first fallback)", file=sys.stderr)
                    answer = self._try_fireworks(entry, item, retry_model).answer
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

    # D55: when the local budget can exhaust mid-batch (slow grading
    # hardware), the tasks processed EARLY get local attempts and the tail
    # goes straight to paid remote -- so process cheapest local categories
    # first to maximize keeps-per-second before exhaustion. Output order is
    # restored to input order afterwards; the {task_id, answer} contract is
    # untouched. Categories not listed (code needs example-proof, remote
    # tiers) sort last -- their local attempts are longest/least likely to
    # keep, or they never attempt local at all.
    _LOCAL_COST_ORDER = {"sentiment": 0, "entity_extraction": 1,
                         "summarization": 2, "general": 3, "knowledge": 3,
                         "math": 4}

    def route_all(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            order = sorted(
                range(len(tasks)),
                key=lambda i: self._LOCAL_COST_ORDER.get(
                    self.classifier.classify(_task_prompt(tasks[i], i)).category
                    if self._safe_prompt(tasks[i], i) is not None else "", 9))
            results_by_pos = {i: self.route_task(tasks[i], i) for i in order}
            return [results_by_pos[i] for i in range(len(tasks))]
        finally:
            self.close()

    @staticmethod
    def _safe_prompt(task: Dict[str, Any], idx: int) -> Optional[str]:
        try:
            return _task_prompt(task, idx)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Draft policy generator, fed by Step 2's probe-local + bakeoff outputs.
# ---------------------------------------------------------------------------

def generate_policy(local_viability: Dict[str, LocalViability],
                    bakeoff_ranking: Dict[str, List[ModelCategoryRanking]],
                    max_tokens: int = 512, prompt_template: str = "default",
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
