#!/usr/bin/env bash
# Rebuild every Stage 1 result from the frozen inputs (README Phase 12). CPU only, no GPU, no model calls.
#
#   ./reproduce_all.sh            # rebuild results/stage1/ and results/safety/, then compare with the previous copy
#   ./reproduce_all.sh --no-check # rebuild only
#
# Inputs (never changed by this script): benchmark/ (tasks, frozen pool v4 with its output pickles), data/,
# kaggle_outputs/ (LLM generations and parser outputs downloaded from Kaggle), configs/baseline2_v4.yaml.
# The pool itself is not rebuilt (eval/build_pool.py refuses to overwrite a frozen pool); its checksums are verified.
# With the check on, the previous results/stage1/ is copied aside first and every result file is compared afterwards;
# run_info.json files (they record run times and paths) are left out of the comparison.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
CHECK=1
[[ "${1:-}" == "--no-check" ]] && CHECK=0

step() { echo; echo "=== $*"; }

step "0. tests (verifier, sandbox, data checksums, reference contracts)"
$PY -m pytest -q -x tests/test_data_checksums.py tests/verifier tests/sandbox/test_ast_guard.py

step "1. frozen pool v4: output pickles match manifest.json"
$PY - <<'EOF'
import hashlib, json, pathlib
m = json.loads(pathlib.Path("benchmark/pool/manifest.json").read_text())
bad = [f for f, h in m["sha256"].items()
       if hashlib.sha256(pathlib.Path("benchmark/pool", f).read_bytes()).hexdigest() != h]
assert not bad, f"pool files changed: {bad[:5]}"
print(f"{len(m['sha256'])} pool files OK")
EOF

if [[ $CHECK == 1 ]]; then
  PREV=$(mktemp -d)
  cp -r results/stage1 "$PREV/stage1"
  [[ -d results/safety ]] && cp -r results/safety "$PREV/safety"
  echo "previous results copied to $PREV"
fi

step "2. Baseline 1 by dataset"
$PY eval/baseline1_by_dataset.py
step "3. Baseline 2 (conventional checks; thresholds re-tuned on dev, configs/baseline2_v4.yaml)"
$PY eval/baseline2.py
step "4. intent verifier with gold contracts (oracle + change-only / preserve-only)"
$PY eval/run_intent_oracle.py
step "5. every downloaded parser run (Baseline 3, size ablation, other prompts, fine-tuned if present) + summary tables"
$PY eval/score_parsers.py
step "5b. Stage 1 recall by error kind (labels: benchmark/annotation/error_kinds_pool_v4.csv)"
$PY eval/error_kind_breakdown.py > /dev/null && echo "results/stage1/summary/error_kinds.md"
step "6. safety table (red-team samples through every execution path)"
$PY eval/safety_table.py > /dev/null && echo "results/safety/redteam_table.md"

if [[ $CHECK == 1 ]]; then
  step "7. compare with the previous results"
  if diff -rq --exclude=run_info.json "$PREV/stage1" results/stage1; then
    echo "results/stage1: identical to the previous copy (run_info.json not compared)"
  else
    echo "results/stage1 DIFFERS from the previous copy (kept in $PREV/stage1); inspect with: diff -r $PREV/stage1 results/stage1"
    exit 1
  fi
  if [[ -d "$PREV/safety" ]]; then
    $PY - "$PREV/safety/redteam_table.json" results/safety/redteam_table.json <<'EOF'
import json, sys
strip = lambda r: {k: v for k, v in r.items() if k != "seconds"}
a, b = (json.load(open(p)) for p in sys.argv[1:])
same = [strip(x) for x in a["samples"]] == [strip(x) for x in b["samples"]] and a["passed"] == b["passed"]
print("results/safety: identical outcomes" if same else "results/safety: outcomes DIFFER")
sys.exit(0 if same else 1)
EOF
  fi
  rm -rf "$PREV"
fi
if [[ -f results/stage2_agent/agent_full/runs.jsonl ]]; then
  step "8. agent test-run metrics (from the downloaded runs; no model calls)"
  $PY eval/agent_metrics.py results/stage2_agent test | tail -4
fi
if ls results/stage2/*/runs.jsonl > /dev/null 2>&1; then
  step "9. Stage 2 metrics (from the downloaded runs)"
  $PY eval/stage2_metrics.py results/stage2 test | head -12
fi
echo; echo "done. Tables: results/STAGE1_RESULTS.md (text), results/stage1/summary/metrics_stage1.json, results/safety/redteam_table.md"
