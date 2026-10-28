#!/usr/bin/env bash
# Prints verified=true|false to $GITHUB_OUTPUT (stdout when unset).
# A push commit is verified when a successful pull_request run of $WORKFLOW
# tested a head commit with the same tree (merge commit or squash merge). Uses
# only actions:read and contents:read, and no checkout. Any API error means
# "not verified": the job then runs in full.
set -uo pipefail
out="${GITHUB_OUTPUT:-/dev/stdout}"
verified=false
tree_of() { gh api "repos/$REPO/commits/$1" --jq .commit.tree.sha 2>/dev/null; }
if [ "$EVENT" = push ] && tree="$(tree_of "$SHA")" && [ -n "$tree" ]; then
    # A merge commit's second parent is the PR head: check it first.
    second="$(gh api "repos/$REPO/commits/$SHA" --jq '.parents[1].sha // empty' 2>/dev/null || true)"
    heads="$(gh run list -R "$REPO" --workflow "$WORKFLOW" --event pull_request --status success \
        --limit "${GATE_RUNS:-15}" --json headSha --jq '.[].headSha' 2>/dev/null | awk '!seen[$0]++' || true)"
    if [ -n "$second" ] && grep -qx "$second" <<<"$heads"; then
        heads="$second"$'\n'"$(grep -vx "$second" <<<"$heads")"
    fi
    for head in $heads; do
        if [ "$(tree_of "$head")" = "$tree" ]; then
            verified=true
            echo "::notice::Tree $tree already passed $WORKFLOW on pull request head ${head:0:12}; skipping"
            break
        fi
    done
fi
echo "verified=$verified" >> "$out"
