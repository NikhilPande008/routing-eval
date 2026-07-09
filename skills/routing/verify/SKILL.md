---
name: routing:verify
description: Run the routing project's verification gate — tests, lint, clean-environment smoke, and (once it exists) the container smoke. Use ALWAYS at the end of any session that touched code, before any commit, and before submission. Trigger on "verify", "run checks", "is the build clean", "check for errors", "pre-commit", or "/routing:verify". Nothing is done until this is green.
---

# routing:verify

The gate. Green here is the definition of "done" for a code change — the
build-vs-verify discipline: nothing counts until it has run through the actual
clean path.

## Steps

1. **Tests:**
   ```bash
   PYTHONPATH=. python -m pytest -q
   ```
   Must be all-pass. On failure, report exact `file::test` and the assertion.

2. **Lint** (if configured; skip cleanly if not):
   ```bash
   ruff check routing_eval tests 2>/dev/null || echo "ruff not configured — skip"
   ```

3. **Clean-environment smoke (the real gate).** Build the image and run the
   quickstart inside it:
   ```bash
   scripts/container_smoke.sh
   ```
   This is the submission gate — it proves `pip install .` + the entrypoint + the
   quickstart run on a genuinely clean image. If Docker is unavailable the script
   exits **2**; only then fall back to the venv stand-in below, and state
   explicitly that the container gate did NOT run:
   ```bash
   rm -rf /tmp/venv-verify && python -m venv /tmp/venv-verify
   /tmp/venv-verify/bin/pip install --quiet .
   cd /tmp && /tmp/venv-verify/bin/routing-eval run --dataset standin --n 60 --out /tmp/vr.json
   /tmp/venv-verify/bin/routing-eval frontier --records /tmp/vr.json --accuracy-threshold 0.80 >/dev/null \
     && echo "venv stand-in OK (container gate NOT run)"
   ```
   The venv path installs the package the same way the image does, so it's a real
   stand-in — but it does not exercise the Docker wrapper. Distinguish the two in
   the log.

## Pass criteria

All present steps green. The clean-env gate is satisfied only when
`container_smoke.sh` passes. If it exited 2 (no Docker) and you ran the venv
stand-in instead, say so explicitly in the session log — the venv path does not
prove the container builds, so do not claim clean-env/submission readiness on it
alone.
