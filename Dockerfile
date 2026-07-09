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

# Run as non-root; /tmp stays writable for records/frontier output.
RUN useradd --create-home --uid 10001 appuser
USER appuser

ENTRYPOINT ["routing-eval"]
CMD ["score"]
