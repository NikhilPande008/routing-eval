---
name: routing:frontier
description: Trace the token-vs-accuracy frontier from records.json, locate the operating point for an accuracy floor, and compare gate signals. Use whenever you need to set or re-tune the escalation threshold, pick between gates, check how close a gate is to optimal, decide whether you are gate-limited or capability-limited, or read the cost of hitting an accuracy target. Trigger on "trace the frontier", "what's the operating point", "compare gates", "tune tau", "how many tokens to hit X accuracy", or "/routing:frontier". Free — spends zero tokens.
---

# routing:frontier

Read the frontier off `records.json` and locate where to operate. Pure
arithmetic; run it as often as you like.

## Steps

1. Run, with the accuracy floor (post-kickoff: the real threshold; pre-kickoff:
   your target):
   ```bash
   python -m routing_eval.cli frontier --records records.json \
       --accuracy-threshold 0.85 --csv-out frontier.csv
   ```
   Add `--signals informative random anti` (or the real gate names) to restrict
   the comparison; default compares all signals present.

2. Read the report per gate:
   - **all-local / all-remote** — the two baselines.
   - **ceiling** — max accuracy ANY per-item router can reach on this set. If it's
     below the floor, no router can win here; the set is capability-bound.
   - **operating point** — min remote tokens to clear the floor with this gate,
     plus escalation rate and margin. Margin near 0 means you're on the aggressive
     edge (good, but watch eval-set variance).
   - **oracle** — min tokens with perfect knowledge (exact for binary scores,
     `approx` for graded).
   - **gate efficiency** = oracle / gate (≤ 1). The diagnostic.

## Interpretation (the whole point)

- **Low gate efficiency** → the confidence signal is leaving tokens on the table.
  Improve the gate (better signal, better calibration), not the model.
- **Oracle infeasible / ceiling < floor** → you are capability-limited. Improve the
  model (fine-tune, swap local model, escalate more), not the gate.
- **all-remote below the floor** but a gate clears it → selective routing is
  beating brute escalation by avoiding items where remote is worse than local.
- Pick the gate whose frontier dominates, then set τ at its operating point plus a
  small variance-sized safety margin.

## Output

Console report + `frontier.csv` (plot it if you want the full curve). Record the
chosen gate + τ + the numbers in DECISIONS.md — that's a backed performance claim.
