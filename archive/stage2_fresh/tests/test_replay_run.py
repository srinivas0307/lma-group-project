"""Regression checks of the v2 fixes against (a) all gold contracts and (b) a real v1 run's outputs."""
import sys, json, glob, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import harness
from test_pipeline_mock import load_variant
ns = load_variant("selective_facts")
tasks = {t["id"]: t for t in ns["ALL_TASKS"]}
problems = []

# (a) gold contracts must not trigger any hard issue (else the checks would reject correct contracts)
for t in ns["ALL_TASKS"]:
    c, _ = ns["normalize_contract"](t["gold_contract"])
    hard, soft = ns["contract_issues"](c, ns["load_before"](t))
    if hard: problems.append(("gold-hard", t["id"], hard))
    if soft: print("gold soft (info):", t["id"], soft)
print("gold contracts with hard issues:", sum(p[0] == "gold-hard" for p in problems))

# (b) replay a real run if present
RUN = os.environ.get("RUN_DIR")
if RUN:
    R = {json.load(open(f))["id"]: json.load(open(f)) for f in glob.glob(f"{RUN}/tasks/*.json")}
    flag = {"correct_accepted": [], "incorrect_accepted": [], "withheld": []}
    for tid, r in R.items():
        c = r["parser"].get("contract")
        if not c: continue
        hard, soft = ns["contract_issues"](c, ns["load_before"](tasks[tid]))
        if hard: flag[r["outcome"]].append((tid, hard[0][:150]))
    for k, v in flag.items():
        print(f"\nv1 {k}: {len(v)} would now get hard feedback")
        for x in v: print("   ", x)
    # code extraction on every recorded reply that v1 could not use
    fixed = 0
    for r in R.values():
        for m in (r.get("generation", {}).get("messages", []) + r.get("repair", {}).get("messages", [])):
            if m["role"] == "assistant" and "```" in m["content"]:
                code = ns["extract_code"](m["content"])
                if code and not ns["format_problem"](code) and code.splitlines()[0].strip() == "python": problems.append(("python-line", r["id"]))
    print("titanic_dedup_002 extracted:\n", ns["extract_code"](R["titanic_dedup_002"]["generation"]["messages"][-1]["content"])[:200])

for tid in ("taxi_miss_005", "titanic_miss_004", "airbnb_miss_004"):
    print("facts", tid, ns["compute_facts"](tasks[tid]["request"], ns["load_before"](tasks[tid])))
if ns["compute_facts"](tasks["taxi_miss_005"]["request"], ns["load_before"](tasks["taxi_miss_005"])): problems.append(("mean-verb", ""))
print("\nPROBLEMS:", problems)
