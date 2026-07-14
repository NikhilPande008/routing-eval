#!/usr/bin/env bash
# Build + push a SUBMITTABLE routing-eval image to Docker Hub in ONE direct step.
#
# Why this script exists (root-caused in DECISIONS.md D43): this machine's Docker
# Desktop has the containerd image store enabled. With containerd, any
# `docker buildx build --load` followed by `docker push` re-encodes the image to
# an OCI manifest, which the hackathon grader's pull backend rejects with
# PULL_ERROR. The fix is a DIRECT push from BuildKit with oci-mediatypes=false,
# so the registry gets the classic Docker v2 manifest. This script does exactly
# that, then VERIFIES the manifest format and prints the tag+digest to submit.
#
# Usage:
#   scripts/push_submission.sh <tag>
#   scripts/push_submission.sh submission-d44
#   REPO=docker.io/nikhilpande/routing-eval scripts/push_submission.sh submission-d44
#
# The <tag> must be a NEW immutable tag -- NEVER reuse a known-good rollback tag
# (submission-17of19, submission-amd64, ...). The script refuses if the tag
# already exists on the registry, to protect your rollback points.
#
# Pushing a new tag does NOT change what the grader pulls -- repointing the
# submission form at this tag+digest is a separate, manual step (your call).
#
# Exit codes: 0 = pushed AND verified classic-v2; 2 = docker/preflight problem;
# other non-zero = build, push, or manifest verification FAILED (do NOT submit).
set -euo pipefail

REPO="${REPO:-docker.io/nikhilpande/routing-eval}"
PLATFORM="${PLATFORM:-linux/amd64}"
cd "$(dirname "$0")/.."

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  echo "usage: $0 <new-immutable-tag>   (e.g. submission-d44)" >&2
  exit 2
fi
REF="$REPO:$TAG"

# --- FROZEN-TAG PUSH GUARD (hardcoded, updated 2026-07-14) -----------------
# Clarified 2026-07-14, user-confirmed: "frozen" scopes to the two ORIGINAL
# rollback pointer tags specifically -- submission-amd64 (the live submission
# pointer) and submission-17of19 (the durable 17/19 rollback) -- not the
# whole repo. New immutable tags (submission-gemma-2/3/4, ...) have been
# pushed here since, confirmed as the intended workflow (no separate Gemma
# evaluation track -- a submission is just graded on whatever the container
# actually does, so relabeling which config sits on a given tag is not a
# disclosure concern). This guard blocks overwriting those two named tags
# only; the generic "refuse to overwrite an EXISTING tag" check below still
# protects every other tag from accidental reuse.
_FROZEN_TAGS=("submission-amd64" "submission-17of19")
if [[ "$REPO" == "docker.io/nikhilpande/routing-eval" ]]; then
  for _frozen in "${_FROZEN_TAGS[@]}"; do
    if [[ "$TAG" == "$_frozen" ]]; then
      echo "PUSH GUARD: refusing to push to FROZEN rollback tag '$REPO:$TAG'." >&2
      echo "  This is one of the two protected pointers (${_FROZEN_TAGS[*]})." >&2
      echo "  Push a new tag instead." >&2
      exit 2
    fi
  done
fi

# --- preflight -------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "push_submission: docker not installed." >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "push_submission: docker daemon not reachable." >&2
  exit 2
fi
if ! docker buildx version >/dev/null 2>&1; then
  echo "push_submission: docker buildx not available." >&2
  exit 2
fi
# Must be logged in to push. `docker login` with no creds cached will fail here.
if ! docker buildx imagetools inspect "$REPO:latest" >/dev/null 2>&1 \
   && ! grep -q "auths" ~/.docker/config.json 2>/dev/null; then
  echo "push_submission: not logged in to a registry. Run 'docker login' first." >&2
  exit 2
fi

# --- protect rollback tags: refuse to overwrite an existing tag -----------
if docker buildx imagetools inspect "$REF" >/dev/null 2>&1; then
  echo "push_submission: tag '$TAG' ALREADY EXISTS on $REPO." >&2
  echo "  Refusing to overwrite -- pick a new tag so rollback points stay intact." >&2
  exit 2
fi

# --- direct build + push (no --load, no separate docker push) -------------
echo "==> building + pushing $REF ($PLATFORM), direct-to-registry, classic v2 manifest"
docker buildx build \
  --platform "$PLATFORM" \
  --provenance=false --sbom=false \
  --output "type=image,name=$REF,oci-mediatypes=false,push=true" \
  .

# --- verify the manifest is classic Docker v2, NOT OCI --------------------
echo "==> verifying manifest media type"
MEDIA_TYPE="$(docker buildx imagetools inspect "$REF" --raw \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['mediaType'])")"
EXPECTED="application/vnd.docker.distribution.manifest.v2+json"
if [[ "$MEDIA_TYPE" != "$EXPECTED" ]]; then
  echo "push_submission: WRONG manifest type: $MEDIA_TYPE" >&2
  echo "  Expected: $EXPECTED" >&2
  echo "  This WILL cause PULL_ERROR on the grader. Do NOT submit this tag." >&2
  exit 3
fi
echo "    manifest OK: $MEDIA_TYPE"

# --- report the tag + digest to paste into the submission form ------------
DIGEST="$(docker buildx imagetools inspect "$REF" \
  | awk '/^Digest:/ {print $2; exit}')"
echo
echo "==> READY TO SUBMIT"
echo "    tag:    $REF"
echo "    digest: $DIGEST"
echo "    platform: $PLATFORM"
echo
echo "Next (manual, your call): repoint the submission form at the tag+digest above."
echo "This push did NOT change what the grader currently pulls."
