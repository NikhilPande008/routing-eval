# Routing agent container.
#
# Runtime footprint is the base Python image plus this package -- ZERO third-party
# runtime dependencies. setuptools/wheel are pulled only at BUILD time to build the
# package; nothing third-party is installed for runtime.
#
# ENTRYPOINT is the CLI. Default CMD is the scoring path (`score`, per the
# official Track 1 guide, DECISIONS.md D20): reads /input/tasks.json, answers
# each task with one Fireworks call (env: FIREWORKS_BASE_URL /
# FIREWORKS_API_KEY / ALLOWED_MODELS), writes /output/results.json, exits 0.
# Override CMD (e.g. `run`, `frontier`, `--help`) for the eval/calibration CLI.
#
# Pin by digest for full reproducibility if desired:
#   FROM python:3.12-slim@sha256:<digest>
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy only what the package needs to build + run (see .dockerignore).
COPY pyproject.toml README.md ./
COPY routing_eval ./routing_eval

# Install the package itself. No runtime deps are declared, so this pulls only the
# build backend, not third-party runtime libraries. The import check fails the
# build early if anything is broken.
RUN pip install . && python -c "import routing_eval; print('build import OK')"

# Create the scoring mount points so an evaluator dry-run without bind mounts
# still exits cleanly. The container intentionally runs as root: on Linux
# graders, /output is commonly a root-owned bind mount, and a non-root user can
# crash while writing /output/results.json even when the entrypoint is correct.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /input /output \
    && chown -R appuser:appuser /input /output

ENTRYPOINT ["routing-eval"]
CMD ["score"]
