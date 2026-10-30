"""Build the R / C / T agent scenarios from a frozen pool (README Phase 4 task 5; guide §11).

    # current version rct_v4 (used by the runner; the defaults): pool v4, 1 wrong + 1 crashed candidate per task,
    # T faults on the agent's first tool call (rct_v3: same scenarios, faults on a named tool)
    .venv/bin/python eval/build_agent_scenarios.py --out /tmp/rct_check.jsonl     # rebuild elsewhere to compare
    # earlier versions: rct_v2 (pool v3, same settings) and rct_v1 (pool v2, 3 per task), in archive/agent_scenarios/

    R  repairable: a wrong LLM candidate (wrong output, or crashed). Success = the agent ends with ACCEPT and the
       final output equals the reference output (judged by the evaluator, never by the verifier).
    C  already correct: a correct candidate. Success = ACCEPT with no repair; any LLM/tool call is unnecessary.
    T  tool fault: an R scenario plus one injected tool fault. Success as for R; also measures recovery.

Sampling (deterministic, seed 0), per split: R = up to --r-per-task (default 1) runnable-wrong candidates per task + up to 1 crashed one;
C = 1 correct candidate per task; T = a stratified sample of R with a fault on the first use of a tool, cycling
through (tool, kind) pairs. Scenario files hold candidate ids and code only; reference code and labels stay in the
pool and are read only by the evaluator.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAULTS = [("diff_datasets", "error"), ("run_verifier", "timeout"), ("profile_column", "malformed"),
          ("inspect_dataset", "error"), ("run_transformation", "timeout"), ("diff_datasets", "malformed")]
# rct_v4: the fault hits the agent's first tool call, whatever the tool ("*"). With a named tool (rct_v1-v3) the
# fault fired in only 1 of 6 dev T scenarios (agent dev run 4), because the agent mostly used other tools.
FAULTS_ANY = [("*", "error"), ("*", "timeout"), ("*", "malformed")]


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def scenario(sid, cat, c, expected, faults=()):
    return dict(id=sid, category=cat, split=c["split"], task_id=c["task_id"], dataset=c["dataset"],
                transformation_type=c["transformation_type"], candidate_id=c["candidate_id"],
                initial_code=c["code"] or "", initial_status=c["status"], expected_outcome=expected,
                faults=[dict(tool=t, on_call=1, kind=k) for t, k in faults])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "benchmark" / "pool" / "pool_v4.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark" / "agent_scenarios" / "rct_v4.jsonl")
    ap.add_argument("--r-per-task", type=int, default=1)
    ap.add_argument("--t-count", type=int, default=24, help="T scenarios on test (dev gets a quarter)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fault-target", choices=["any", "named"], default="any",
                    help='any: fault on the first tool call of any tool (rct_v4); named: a fixed tool (rct_v1-v3)')
    ap.add_argument("--force", action="store_true", help="overwrite an existing scenario file")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; scenario files are frozen once used. Use --force only if you mean it.")
    rng = random.Random(args.seed)
    pool = [c for c in load_jsonl(args.pool) if c["parse_ok"]]
    out = []
    for split in ("dev", "test"):
        by_task = defaultdict(list)
        for c in pool:
            if c["split"] == split:
                by_task[c["task_id"]].append(c)
        R, C = [], []
        for tid in sorted(by_task):
            cs = sorted(by_task[tid], key=lambda c: c["candidate_id"])
            wrong = [c for c in cs if c["ran_ok"] and not c["is_correct"]]
            crash = [c for c in cs if not c["ran_ok"] and not c["blocked"]]
            correct = [c for c in cs if c["is_correct"]]
            R += rng.sample(wrong, min(args.r_per_task, len(wrong))) + rng.sample(crash, min(1, len(crash)))
            C += rng.sample(correct, min(1, len(correct)))
        for i, c in enumerate(R, 1):
            out.append(scenario(f"R_{split}_{i:03d}", "R", c, "ACCEPT_CORRECT"))
        for i, c in enumerate(C, 1):
            out.append(scenario(f"C_{split}_{i:03d}", "C", c, "ACCEPT_NO_ACTION"))
        n_t = args.t_count if split == "test" else max(2, args.t_count // 4)
        by_type = defaultdict(list)
        for c in R:
            by_type[c["transformation_type"]].append(c)
        picks = []
        while len(picks) < min(n_t, len(R)):                  # round-robin over types
            for t in sorted(by_type):
                if by_type[t] and len(picks) < n_t:
                    picks.append(by_type[t].pop(rng.randrange(len(by_type[t]))))
        for i, c in enumerate(picks, 1):
            faults = FAULTS_ANY if args.fault_target == "any" else FAULTS
            out.append(scenario(f"T_{split}_{i:03d}", "T", c, "ACCEPT_CORRECT", [faults[(i - 1) % len(faults)]]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(s) + "\n" for s in out))
    from collections import Counter
    print(dict(Counter((s["split"], s["category"]) for s in out)))
    print("R test by type:", dict(Counter(s["transformation_type"] for s in out if s["category"] == "R" and s["split"] == "test")))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
