---
name: routing:run
description: Generate the eval records.json by running the local and remote models over a dataset and scoring both. Use whenever you need to (re)generate records for the routing harness, refresh the eval set, produce the input the frontier tracer consumes, or re-run after changing models, the dataset, or the scorer. Trigger on "run the harness", "generate records", "rebuild records.json", "produce eval records", or "/routing:run". This is the ONLY step that spends remote tokens — run it deliberately.
---

# routing:run

Produce `records.json` — one Record per eval item, holding both model scores, the
remote token cost, and every candidate gate signal. This is the expensive half of
record-then-replay; everything downstream (frontier, τ-tuning, gate comparison) is
free once this file exists.

## Preconditions

- Decide the dataset. Pre-kickoff: the synthetic stand-in. Post-kickoff: the real
  eval set wired into `datasets.py` with the revealed scorer set on each Item.
- Decide the models:
  - **Mock (default, pre-reveal / no keys):** free, deterministic. Use for testing
    the harness and the workflow.
  - **Real (P2):** vLLM-local runner + Fireworks remote runner. **This spends real
    remote tokens on the whole dev set once.** Confirm you mean to before running.

## Steps

1. Confirm mock vs real and the accuracy target you'll later evaluate against.
2. Run:
   ```bash
   # mock (free, deterministic)
   python -m routing_eval.cli run --dataset standin --n 150 --out records.json

   # real (P2; spends tokens) — via the P2 runner entrypoint, same schema out
   ```
   Tunable mock knobs: `--local-competence EASY BORDER HARD`, `--calib-noise`
   (lower = better-calibrated gate), `--remote-competence`.
3. Sanity-check the output before trusting it:
   ```bash
   python -c "import json; r=json.load(open('records.json')); \
     print('records', len(r)); print('signals', list(r[0]['confidences'])); \
     assert all(0<=x['local_score']<=1 and 0<=x['remote_score']<=1 for x in r); \
     assert all(x['remote_total_tokens']==x['remote_prompt_tokens']+x['remote_completion_tokens'] for x in r); \
     print('schema OK')"
   ```

## Output

`records.json`. Cache it. If it came from a real run, do not regenerate casually —
you re-pay the token cost. Then run `routing:frontier`.

## Guardrail

A real run's token spend is the single largest controllable cost in the project.
State the estimated dev-set token cost in the session log before a real run.
