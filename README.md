# Token-Efficient Routing Agent — AMD Developer Hackathon (Track 1)

A cost-aware LLM router that answers each task with the cheapest source that
can get it right: a **bundled local model first**, gated by **deterministic
validators**, escalating to a **frontier remote model only on proven doubt**.

Built for AMD Developer Hackathon ACT II, Track 1, where entries are graded by
an external LLM judge and **ranked by remote token count after clearing an 80%
accuracy gate**. Local-model tokens count as zero toward the score.

**Runtime dependencies: none** (pure Python standard library). The container
runs on a clean environment with no third-party runtime installs. `pytest` is
dev-only.

---

## TL;DR — what shipped and how it scored

| Config | Accuracy (live) | Remote tokens (live) | What it is |
|---|---|---|---|
| **Final kimi line** | **89.5% (17/19)** | **4,390** | bundled 1.5B local tier + kimi escalation, validator-gated |
| High-accuracy remote line | 94.7–100% | 5,500–6,100 | mostly-remote; used to probe the accuracy ceiling |

Every shipped config cleared the 80% gate across 11 live submissions. Token
spend was driven from an inherited 5,139 down to a stable 3,600–4,400 band,
while accuracy held or improved. The one target that proved **structurally
unreachable** — sub-2,000 tokens *and* >80% — is explained under
[Lessons](#engineering-lessons-the-honest-part).

---

## The problem

- **Scored by an external LLM judge**, not a metric we control. Rubric grades
  *intent and format*, not string match.
- **Ranked by tokens once you pass the 80% gate.** Accuracy above the gate buys
  no rank — so the game is *minimum tokens at ≥80%*, not maximum accuracy.
- **Only remote (Fireworks) tokens are metered.** A correct answer from a model
  running inside the container is free.
- Hard constraints: 4 GB RAM / 2 vCPU grading VM, 10-minute total budget,
  30 s/request, container reads `/input/tasks.json` → writes
  `/output/results.json`.

## Architecture: a validator-gated cascade

```
task ─▶ classify category ─▶ policy lookup
                                  │
                    ┌─────────────┴─────────────┐
              tier: local                   tier: fireworks
                    │                             │
        bundled model answers            remote model answers
                    │                     (kimi / frontier)
         deterministic validators                │
         + self-consistency check                │
                    │                             │
         pass ──▶ keep (0 tokens)                 │
         fail ──▶ escalate ────────────────────▶ ┘
```

The design rests on three ideas:

1. **Cascade, not a pre-router.** Local tokens are free, so *attempting* local
   first is costless and yields richer signals (validator outcomes,
   self-consistency) than deciding before generating.
2. **The intelligence lives in the gate.** A local answer is kept only if it
   passes a deterministic, stdlib-only shape check for its category. Any doubt
   escalates. The asymmetry is deliberate: **a rejected local answer only costs
   tokens; an accepted bad answer costs accuracy** — so the checks are strict
   and over-escalation is the cheap failure mode.
3. **Graceful degradation everywhere.** No local model, a wedged server, a
   transient remote error, or slow hardware all fall back to the proven remote
   path. The container never crashes on a single task; a failed task still emits
   a valid (empty-if-necessary) response.

### The validators (`routing_eval/localcheck.py`)

Deterministic checks that gate local answers, tuned to the official rubric:

- **Sentiment** — label present and from the allowed set; a mixed review must
  acknowledge *both* sides (a one-sided justification fails regardless of label).
- **NER** — `entity -- TYPE` lines using only the official four labels
  (PERSON/ORGANIZATION/LOCATION/DATE); every entity verified verbatim in the
  source; a date split across two lines is caught and escalated.
- **Summarization** — exact sentence/bullet counts and per-bullet word caps
  enforced; summary must be shorter than the source.
- **Math / knowledge** — two-sample self-consistency: a second sample at a
  different temperature must agree (exact final-number match for math) or the
  task escalates. Content the validators can't check gets this instead.

### Robustness plumbing (`routing_eval/policy.py`)

- **Batch-level time governor** — bounds total local wall-clock so slow
  hardware degrades to remote-only rather than blowing the 10-minute budget.
- **Cheapest-first ordering** — processes local-eligible tasks in order of
  expected cost so the most keeps land before the governor exhausts.
- **Escalation retry** — a blank/failed remote answer retries with a known-good
  model, never an undeployed one (a real bug that once shipped an empty answer).
- **Speed probe** — a startup latency check disables the local tier entirely on
  hardware too slow to beat the per-request timeout.

## Repository layout

```
routing_eval/
  classify.py     category detectors (deterministic, phrase-level)
  policy.py       the router: classify → local/remote → {task_id, answer}
  localcheck.py   deterministic validators gating local answers
  prompts.py      per-category system prompts
  llm/            OpenAI-compatible client + local/remote runners
  conformance.py  scoring entrypoint: /input/tasks.json → /output/results.json
  frontier.py     token-vs-accuracy analysis harness (record-then-replay)
  scorers.py      pluggable accuracy scorers
scripts/          battery/diagnostic tooling, container smoke tests
tests/            153 tests (routing, validators, frontier, conformance)
Dockerfile        python:3.12-slim + this package (+ optional bundled model)
```

The container is configurable by build arg / env so **one image serves
multiple model lines** — the policy file and remote model are selected at build
or run time without a code change.

## Running it

```bash
# tests
PYTHONPATH=. python -m pytest -q

# build the scoring container
docker build -t routing-eval .

# run the scoring contract locally (stub Fireworks server, no key needed)
bash scripts/conformance_smoke.sh

# real endpoint (requires FIREWORKS_BASE_URL / FIREWORKS_API_KEY / ALLOWED_MODELS)
set -a && source .env && set +a
REAL_FIREWORKS=1 bash scripts/conformance_smoke.sh
```

At grading, the harness injects `FIREWORKS_BASE_URL`, `FIREWORKS_API_KEY`, and
`ALLOWED_MODELS`; the container reads `/input/tasks.json`, writes
`/output/results.json`, and exits 0. A personal key is for local dev only and
is never bundled into the image.

## Engineering lessons (the honest part)

The competition was an adversarial, slow-feedback optimization (grading turned
around in tens of minutes to hours). The durable takeaways:

- **Offline evals predict little about live results.** Two structurally
  different changes both passed every offline check (a lenient LLM judge-proxy,
  throttled container gates, the full test suite) and both failed the real gate
  at the *identical* 73.7%. Offline signal is directional; the live judge is the
  only ground truth.
- **Isolated single-variable changes beat bundled ones.** When a bundled change
  fails and per-task results aren't exposed, you can't localize the cause.
  Every subsequent change shipped as one attributable variable, each
  live-validated before the next — which is what made regressions reversible.
- **Draw-to-draw variance is large.** Each submission drew a fresh randomized
  task set; token counts swung ±800–1,000 and accuracy by a task or two on
  identical configs. Single-run comparisons between near-identical configs are
  unreliable.
- **Small local models have specific, catchable failure modes.** A 1.5–3B model
  fails multi-step percentage arithmetic, splits multi-token date entities, and
  gives one-sided justifications on mixed-sentiment reviews. Deterministic
  validators + self-consistency convert an unreliable free tier into a safe one
  via escalate-on-any-doubt — but they can't make it *accurate*, only *safe*.
- **There is a real capability ceiling.** Sub-2,000 tokens *and* >80% requires a
  local model strong enough to answer most tasks correctly for free. A 1.5B
  isn't that model; a larger one failed on latency under the 2-vCPU budget. The
  two targets are in direct conflict for the hardware on hand — a genuine
  ceiling, not a tuning gap.
- **Know where your code actually runs.** The grader injects its own
  credentials, so a private on-demand model deployment in a participant's
  account is unreachable at grading — it silently falls back to the shared
  models. Verify the execution environment before building around a dependency.

## Notes

- Pure-stdlib package; the only non-Python bundled asset (an optional local
  model + `llama.cpp` CPU server) ships as image layers, not as a Python
  dependency.
- The `frontier.py` harness implements a *record-then-replay* method: run local
  and remote once over a dev set, then sweep the escalation threshold as pure
  arithmetic — comparing gates and re-tuning at zero token cost.
- No secrets, keys, or account identifiers are committed to this repository.
