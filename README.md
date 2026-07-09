# routing-eval (P1)

The measurement instrument for a token-vs-accuracy cascade router. It answers
the one question you cannot answer by intuition: **given an accuracy floor, what
is the fewest remote tokens you can spend, and which gate gets you there?**

Built for the AMD Developer Hackathon ACT II, Track 1 (Hybrid Token-Efficient
Routing Agent), where the score is remote token count + accuracy on a hidden
standardized environment. Local-model tokens count as zero.

Runtime dependencies: **none** (pure standard library). `pytest` is used only
for the test suite. This matters: the submission must run on a clean container.

---

## The one idea: record-then-replay

The expensive step and the tuning step are decoupled.

1. **Run once (costs tokens/compute):** run the local model and the remote model
   on every eval item, score both, and write a `records.json`. Each record holds
   everything needed to simulate *any* escalation threshold: both scores, the
   remote token cost, and one or more gate confidence signals.
2. **Sweep forever (free):** the frontier tracer reads `records.json` and sweeps
   the escalation threshold τ. This is pure arithmetic — no model calls. Comparing
   ten gates or re-tuning τ costs zero tokens.

Consequence for the competition: you spend remote tokens on your whole dev set
*once* to calibrate. In production you escalate only the low-confidence tail, so
you spend far fewer. Cache `records.json` and never re-pay.

Consequence for building now: the entire frontier logic is testable today with
mock models. P2 swaps real models into the same record schema; nothing
downstream changes.

## Why a cascade (not a predictive pre-router)

Because local tokens are free here, always attempting local first is costless and
yields richer signals (logprobs, self-consistency) than a pre-router that decides
before generating. A pre-router only helps as a latency optimization *if* the
scoring env imposes a per-item time budget. Until we learn otherwise at kickoff,
cascade strictly dominates.

The objective is a **constrained** optimization: minimize remote tokens subject
to accuracy ≥ floor. The floor is a constraint, not the objective. The optimal
operating point therefore sits just above the floor plus a safety margin sized to
eval-set variance — the aggressive edge. You cannot find that edge without the
frontier, which is why this harness is priority one.

## Quickstart

```bash
# 1. generate records (mock models; deterministic)
python -m routing_eval.cli run --dataset standin --n 150 --out records.json

# 2. trace the frontier at an accuracy floor; compare all gate signals
python -m routing_eval.cli frontier --records records.json \
    --accuracy-threshold 0.85 --csv-out frontier.csv

# tests
PYTHONPATH=. python -m pytest -q
```

The `frontier` command reports, per gate signal: all-local and all-remote
baselines, the **union ceiling** (the maximum accuracy any per-item router can
reach — you cannot be right on an item unless at least one model is right), the
**operating point** (min tokens to clear the floor with this gate), the
**oracle** (min tokens with perfect knowledge), and **gate efficiency**
(oracle / gate ≤ 1). Efficiency is the diagnostic: if it is low, improve the
confidence signal; if even the oracle can't clear the floor, you are
capability-limited and must improve the model, not the gate.

## What the harness reads: the P1 ↔ P2 contract

`routing_eval/schema.py` defines `Record` — the single boundary between this
harness and the router. P2's job is to produce these records from real models.
Key fields:

- `local_score`, `remote_score` — floats in [0,1] from the task's scorer.
- `remote_total_tokens` — what the competition counts if the item escalates.
- `confidences: {name -> float}` — one score per candidate gate. **Convention:
  higher = keep local** (less likely to escalate). A native signal pointing the
  other way (entropy, perplexity) must be negated before storage.

`routing_eval/runner.py` defines the `ModelRunner` protocol P2 implements, plus a
reference sketch of the token-minimal Fireworks call (tight `max_tokens`, `stop`
sequences, `reasoning_effort="none"`, `logprobs` for the gate).

## Components

- `frontier.py` — the tracer, oracle, operating point, gate efficiency. The core.
- `scorers.py` — pluggable accuracy scorers (numeric, exact, multiple_choice,
  token_f1, json_match, code_tests). Add the real metric here at kickoff.
- `datasets.py` — synthetic, correct-by-construction stand-ins (math /
  classification / qa) with a **dense borderline band** — the region where the
  escalate/keep decision breaks and where happy-path testing lies to you.
- `mock.py` — mock runners with knobs for local competence and gate calibration.
  A validation instrument: it lets the tests assert that a better-calibrated gate
  wins, which is the harness's entire reason to exist.
- `report.py`, `cli.py` — rendering and the two commands.

## Container

The routing app containerizes with zero runtime dependencies — the base
`python:3.12-slim` image plus this package is the whole footprint.

```bash
docker build -t routing-eval .
docker run --rm routing-eval --help
# quickstart inside the image:
docker run --rm --entrypoint sh routing-eval -c \
  'routing-eval run --dataset standin --n 60 --out /tmp/r.json && \
   routing-eval frontier --records /tmp/r.json --accuracy-threshold 0.80'
```

`scripts/container_smoke.sh` runs exactly this as the clean-environment gate
(it's what `routing:verify` calls). The default entrypoint is the calibration/eval
CLI; the **scoring-time entrypoint that speaks to the hackathon's scoring harness
is wired at kickoff**, once that interface is known.

## Scope: what is real vs. what is stubbed

Real and tested (16 passing tests):
- Frontier logic, verified against a 4-item case computed by hand: baselines,
  non-monotonic accuracy (escalating a "remote-wrong / local-right" item lowers
  accuracy), operating-point selection, exact oracle for binary scores,
  infeasibility flag, gate efficiency.
- The scorers listed above.
- The full pipeline dataset → mock run → frontier, including the property that a
  lower-noise gate reaches the floor with fewer tokens than a random or
  anti-correlated one.

Stubbed / not yet done:
- **Real model runners (vLLM local + Fireworks remote) are P2.** Only the
  protocol and a doc sketch exist here; no live calls are made.
- `code_tests` executes model-generated code — documented interface only, **not**
  exercised by the stand-in; run it only inside a sandbox.
- The oracle is **exact for binary scores**; for graded scores (token_f1) it is a
  greedy gain-per-token approximation, flagged as `approx` in the report.

What this does **not** claim: that any particular gate or operating point will
work on the real hackathon task. The real task's difficulty distribution, the
revealed models' competence, and the real confidence signals are unknown until
kickoff. The instrument is built and trustworthy; the operating point for the
real task is measured on July 6, not asserted now.
