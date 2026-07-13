#!/usr/bin/env bash
# Build + push the ACCOUNT-2 Gemma-line image in ONE direct step.
#
# This bakes the Gemma policy into the image via build args (POLICY_PATH +
# optional GEMMA_MODEL_ID) so the SAME Dockerfile that builds the kimi line
# builds the Gemma line -- the only difference is these two args. Uses the
# same containerd/OCI fix as push_submission.sh (direct BuildKit push with
# oci-mediatypes=false) so the grader gets a classic Docker v2 manifest and
# never hits PULL_ERROR (DECISIONS.md D43).
#
# Usage:
#   REPO=docker.io/<account2-namespace>/routing-eval scripts/push_account2.sh <tag>
#   REPO=docker.io/acct2/routing-eval GEMMA_MODEL_ID=gemma-4-31b-it \
#       scripts/push_account2.sh submission-gemma-1
#
# REPO is REQUIRED (no default -- you supply the account-2 namespace).
# GEMMA_MODEL_ID is OPTIONAL: unset => the Gemma entries resolve straight to
# the kimi fallback at runtime (accuracy-safe, but NOT a real Gemma run); set
# it to bake the real Gemma model id into the image.
#
# Exit codes: 0 = pushed AND verified classic-v2; 2 = preflight/guard problem;
# other non-zero = build/push/verify FAILED (do NOT submit).
set -euo pipefail

PLATFORM="${PLATFORM:-linux/amd64}"
GEMMA_POLICY_IN_IMAGE="/app/routing_eval/routing_policy.gemma.json"
cd "$(dirname "$0")/.."

REPO="${REPO:-}"
TAG="${1:-}"
if [[ -z "$REPO" || -z "$TAG" ]]; then
  echo "usage: REPO=docker.io/<account2-ns>/routing-eval $0 <new-immutable-tag>" >&2
  echo "  (optionally set GEMMA_MODEL_ID=<id> to bake the real Gemma model)" >&2
  exit 2
fi
REF="$REPO:$TAG"

# --- FROZEN-ACCOUNT PUSH GUARD (hardcoded) ---------------------------------
# Never let the account-2 tooling touch account 1's frozen repo, even by a
# copy-paste REPO mistake.
case "$REPO/" in
  *nikhilpande/routing-eval/*)
    echo "PUSH GUARD: refusing to push to FROZEN account-1 repo '$REPO'." >&2
    echo "  This is the account-2 script. Account 1 is frozen and off-limits." >&2
    exit 2 ;;
esac

# --- preflight -------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "push_account2: docker daemon not reachable." >&2
  exit 2
fi
if ! docker buildx version >/dev/null 2>&1; then
  echo "push_account2: docker buildx not available." >&2
  exit 2
fi
if ! grep -q "auths" ~/.docker/config.json 2>/dev/null; then
  echo "push_account2: not logged in. Run 'docker login' as ACCOUNT 2 first." >&2
  exit 2
fi

# --- protect rollback tags: refuse to overwrite an existing tag ------------
if docker buildx imagetools inspect "$REF" >/dev/null 2>&1; then
  echo "push_account2: tag '$TAG' ALREADY EXISTS on $REPO -- pick a new tag." >&2
  exit 2
fi

# --- build args: the ONLY difference from the kimi line --------------------
BUILD_ARGS=(--build-arg "POLICY_PATH=${GEMMA_POLICY_IN_IMAGE}")
if [[ -n "${GEMMA_MODEL_ID:-}" ]]; then
  BUILD_ARGS+=(--build-arg "GEMMA_MODEL_ID=${GEMMA_MODEL_ID}")
  echo "==> baking GEMMA_MODEL_ID=${GEMMA_MODEL_ID}"
else
  echo "==> GEMMA_MODEL_ID unset: image will fall back to kimi at runtime (accuracy-safe)"
fi

echo "==> building + pushing $REF ($PLATFORM), Gemma policy, direct-to-registry classic v2"
docker buildx build \
  --platform "$PLATFORM" \
  --provenance=false --sbom=false \
  "${BUILD_ARGS[@]}" \
  --output "type=image,name=$REF,oci-mediatypes=false,push=true" \
  .

# --- verify manifest is classic Docker v2 (NOT OCI) ------------------------
echo "==> verifying manifest media type"
MEDIA_TYPE="$(docker buildx imagetools inspect "$REF" --raw \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['mediaType'])")"
EXPECTED="application/vnd.docker.distribution.manifest.v2+json"
if [[ "$MEDIA_TYPE" != "$EXPECTED" ]]; then
  echo "push_account2: WRONG manifest type: $MEDIA_TYPE (expected $EXPECTED)." >&2
  echo "  This WILL cause PULL_ERROR on the grader. Do NOT submit this tag." >&2
  exit 3
fi
echo "    manifest OK: $MEDIA_TYPE"

# --- size check (grader cap is 10GB, D16/D19) ------------------------------
SIZE_BYTES="$(docker buildx imagetools inspect "$REF" --raw \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(l['size'] for l in d.get('layers',[])) + d.get('config',{}).get('size',0))")"
SIZE_MB=$(( SIZE_BYTES / 1000000 ))
echo "==> compressed image size: ~${SIZE_MB} MB (grader cap 10GB)"
if [[ "$SIZE_MB" -gt 10000 ]]; then
  echo "push_account2: image exceeds the 10GB cap." >&2
  exit 3
fi

DIGEST="$(docker buildx imagetools inspect "$REF" | awk '/^Digest:/ {print $2; exit}')"
echo
echo "==> READY TO SUBMIT (account 2)"
echo "    tag:      $REF"
echo "    digest:   $DIGEST"
echo "    platform: $PLATFORM"
echo "    policy:   Gemma (baked POLICY_PATH=${GEMMA_POLICY_IN_IMAGE})"
echo "    gemma id: ${GEMMA_MODEL_ID:-<unset -> kimi fallback>}"
