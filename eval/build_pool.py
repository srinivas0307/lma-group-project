"""Build the frozen candidate pool v1 from the Baseline 1 Kaggle outputs (CPU only, no model needed).

Inputs (unzip baseline1_outputs.zip from the Kaggle run into --b1, e.g. outputs/baseline1/):
    candidates.jsonl   every generation with its sandbox label
    harness.py         the exact sandbox/comparator used on Kaggle
    env.json           run settings (model, prompt hash, pinned versions)

Outputs (in --out, default eval/pool_v1/):
    pool_v1.jsonl              one row per unique (task, code); this is what every system judges
    outputs/<candidate_id>.pkl the candidate's output DataFrame (only for candidates that ran)
    verdicts_baseline1.jsonl   Baseline 1 verdicts in the shared Verdict format
    manifest.json              counts, settings and sha256 of every file

Each candidate is re-executed with the same harness.py. Its label must match the Kaggle label;
any mismatch is reported and makes the script exit with an error, so the pool is never frozen
with labels that can't be reproduced.

Run with the pinned environment (pandas 2.3.3, numpy 2.1.3):
    .venv/bin/python eval/build_pool.py --b1 outputs/baseline1
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINNED = {"pandas": "2.3.3", "numpy": "2.1.3"}
CRASH = {"error", "bad_return", "timeout", "killed", "memory"}

# Runs in a fresh subprocess per candidate: execute, label, pickle the output.
RUNNER = r'''
import json, pickle, sys, time, hashlib
sys.path.insert(0, sys.argv[1])
import harness
import pandas as pd
root, ref_dir, job_path, out_pkl, res_path, cpu_s, mem_gb = sys.argv[2:9]
job = json.load(open(job_path))
df = harness.load_data(root, job["data_file"])
ref = pickle.load(open(f"{ref_dir}/{job['task_id']}.pkl", "rb"))
res = dict(status="error", ran_ok=False, correct=False)
try:
    harness.set_limits(int(cpu_s), float(mem_gb))
except Exception:
    pass
try:
    out = harness.exec_transform(job["code"], job["func_name"], df)
    if not isinstance(out, pd.DataFrame):
        res.update(status="bad_return", error=f"returned {type(out).__name__}, not a DataFrame")
    else:
        ok, reason = harness.outputs_equal(ref, out)
        blob = pickle.dumps(out)
        open(out_pkl, "wb").write(blob)
        res.update(status="ok", ran_ok=True, correct=bool(ok), reason=reason,
                   output_sha256=hashlib.sha256(blob).hexdigest())
except MemoryError:
    res.update(status="memory", error="MemoryError")
except BaseException as e:
    res.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
json.dump(res, open(res_path, "w"), default=str)
'''


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")


def check_env(py):
    v = subprocess.run([py, "-c", "import pandas, numpy; print(pandas.__version__, numpy.__version__)"],
                       capture_output=True, text=True, check=True).stdout.split()
    got = dict(zip(["pandas", "numpy"], v))
    if got != PINNED:
        sys.exit(f"pinned versions required {PINNED}, got {got}. Use .venv/bin/python.")
    return got


def group_candidates(gens):
    """One pool entry per unique (task, normalized code). Unparsed outputs are keyed by raw text."""
    groups = defaultdict(list)
    for g in gens:
        key = g.get("code_hash") or "raw:" + hashlib.sha256(g["raw_output"].encode()).hexdigest()[:16]
        groups[(g["task_id"], key)].append(g)
    pool = []
    for task_id in sorted({k[0] for k in groups}):
        keys = [k for k in groups if k[0] == task_id]
        # stable order: greedy first, then by first sample index
        keys.sort(key=lambda k: min((g["mode"] != "greedy", g["sample_idx"]) for g in groups[k]))
        for n, k in enumerate(keys):
            members = sorted(groups[k], key=lambda g: (g["mode"] != "greedy", g["sample_idx"]))
            first = members[0]
            labels = {(m["ran_ok"], m["correct"]) for m in members}
            pool.append(dict(
                candidate_id=f"{task_id}__c{n:02d}", task_id=task_id, split=first["split"],
                dataset=first["dataset"], transformation_type=first["transformation_type"],
                difficulty=first.get("difficulty"), generator_model=first["model"],
                prompt_version=first["prompt_version"], prompt_hash=first["prompt_hash"],
                gen_ids=[m["gen_id"] for m in members], n_generations=len(members),
                includes_greedy=any(m["mode"] == "greedy" for m in members),
                seeds=sorted({m["seed"] for m in members if m["seed"] is not None}),
                code=first.get("code"), code_hash=first.get("code_hash"), func_name=first.get("func_name"),
                parse_ok=bool(first.get("parse_ok")), blocked=bool(first.get("blocked")),
                kaggle_status=first["status"], kaggle_ran_ok=bool(first["ran_ok"]),
                kaggle_correct=bool(first["correct"]), kaggle_reason=first.get("reason"),
                kaggle_error=first.get("error"), consistent_within_group=len(labels) == 1,
            ))
    return pool


def rerun(cand, args, tmp, out_dir):
    if not cand["parse_ok"] or cand["blocked"]:
        return dict(status="parse_error" if not cand["parse_ok"] else "blocked", ran_ok=False, correct=False)
    job = Path(tmp) / f"{cand['candidate_id']}.json"
    res = Path(tmp) / f"{cand['candidate_id']}.result.json"
    pkl = out_dir / f"{cand['candidate_id']}.pkl"
    job.write_text(json.dumps(dict(task_id=cand["task_id"], data_file=args.data_file[cand["task_id"]],
                                   code=cand["code"], func_name=cand["func_name"])))
    runner = Path(tmp) / "runner.py"
    cwd = Path(tmp) / cand["candidate_id"]
    cwd.mkdir(exist_ok=True)
    try:
        p = subprocess.run([args.python, str(runner), str(args.b1), str(args.root), str(args.b1 / "refs"),
                            str(job), str(pkl), str(res), str(args.cpu), str(args.mem)],
                           capture_output=True, text=True, timeout=args.timeout, cwd=cwd,
                           env=dict(os.environ, PYTHONNOUSERSITE="1"))
    except subprocess.TimeoutExpired:
        return dict(status="timeout", ran_ok=False, correct=False, error=f"no result after {args.timeout}s")
    if not res.exists():
        return dict(status="killed", ran_ok=False, correct=False, error=f"exit {p.returncode}: {p.stderr[-300:]}")
    return json.loads(res.read_text())


def verdict_b1(c):
    """Baseline 1 = verifier 'none': PASS whenever the code ran; anything else is REJECT (repair off)."""
    if c["ran_ok"]:
        v = dict(status="PASS", failures=[], evidence_rows=[])
        outcome = "ACCEPT"
    else:
        v = dict(status="VIOLATION", failures=[dict(kind="execution", status=c["status"], detail=c.get("error"))],
                 evidence_rows=[])
        outcome = "REJECT"
    return dict(system="baseline1", verifier="none", candidate_id=c["candidate_id"], task_id=c["task_id"],
                split=c["split"], verdict=v, outcome=outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", type=Path, required=True, help="unzipped baseline1_outputs folder")
    ap.add_argument("--root", type=Path, default=ROOT, help="project root with data/ and eval/tasks/")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "pool_v1")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--cpu", type=int, default=30)
    ap.add_argument("--mem", type=float, default=4)
    ap.add_argument("--force", action="store_true", help="overwrite an existing pool")
    args = ap.parse_args()
    args.b1, args.root, args.out = args.b1.resolve(), args.root.resolve(), args.out.resolve()

    for f in ["candidates.jsonl", "harness.py", "refs"]:
        if not (args.b1 / f).exists():
            sys.exit(f"missing {args.b1 / f}; unzip baseline1_outputs.zip into --b1")
    if (args.out / "pool_v1.jsonl").exists() and not args.force:
        sys.exit(f"{args.out / 'pool_v1.jsonl'} already exists; the pool is frozen. Use --force only if you mean it.")
    versions = check_env(args.python)

    gens = load_jsonl(args.b1 / "candidates.jsonl")
    for field in ["model", "prompt_hash"]:
        vals = {g[field] for g in gens}
        if len(vals) != 1:
            sys.exit(f"candidates.jsonl mixes several values of {field}: {vals}")
    ids = Counter(g["gen_id"] for g in gens)
    if any(n > 1 for n in ids.values()):
        sys.exit(f"duplicate gen_ids in candidates.jsonl: {[k for k, n in ids.items() if n > 1][:5]}")

    sys.path.insert(0, str(args.b1))
    import harness  # noqa: E402  (the exact Kaggle harness)
    args.data_file = {t["id"]: t["data_file"] for t in harness.load_tasks(str(args.root))}

    pool = group_candidates(gens)
    out_dir = args.out / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "runner.py").write_text(RUNNER)
        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(lambda c: rerun(c, args, tmp, out_dir), pool))

    mismatches = []
    for c, r in zip(pool, results):
        c.update(status=r["status"], ran_ok=bool(r["ran_ok"]), is_correct=bool(r["correct"]),
                 reason=r.get("reason"), error=r.get("error"),
                 output_file=f"outputs/{c['candidate_id']}.pkl" if r["ran_ok"] else None,
                 output_sha256=r.get("output_sha256"))
        if (c["ran_ok"], c["is_correct"]) != (c["kaggle_ran_ok"], c["kaggle_correct"]):
            mismatches.append(dict(candidate_id=c["candidate_id"], kaggle=(c["kaggle_ran_ok"], c["kaggle_correct"]),
                                   local=(c["ran_ok"], c["is_correct"]), local_error=c["error"]))
    inconsistent = [c["candidate_id"] for c in pool if not c["consistent_within_group"]]

    write_jsonl(args.out / "pool_v1.jsonl", pool)
    write_jsonl(args.out / "verdicts_baseline1.jsonl", [verdict_b1(c) for c in pool])

    summary = {}
    for split in sorted({c["split"] for c in pool}):
        S = [c for c in pool if c["split"] == split]
        summary[split] = dict(
            unique_candidates=len(S), ran_ok=sum(c["ran_ok"] for c in S),
            correct=sum(c["is_correct"] for c in S),
            runnable_wrong=sum(c["ran_ok"] and not c["is_correct"] for c in S),
            crashed=sum(c["status"] in CRASH for c in S),
            runnable_wrong_by_type=dict(sorted(Counter(c["transformation_type"] for c in S
                                                       if c["ran_ok"] and not c["is_correct"]).items())),
        )
    manifest = dict(
        pool_version="v1", source=str(args.b1), kaggle_env=json.loads((args.b1 / "env.json").read_text())
        if (args.b1 / "env.json").exists() else None, local_versions=versions,
        n_generations=len(gens), n_candidates=len(pool), summary=summary,
        label_mismatches=mismatches, inconsistent_groups=inconsistent,
        sha256={p.relative_to(args.out).as_posix(): sha256_file(p)
                for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "manifest.json"},
        sha256_candidates_jsonl=sha256_file(args.b1 / "candidates.jsonl"),
        sha256_harness=sha256_file(args.b1 / "harness.py"),
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"{len(gens)} generations -> {len(pool)} unique candidates -> {args.out}")
    for split, s in summary.items():
        print(f"  {split}: {s}")
    if inconsistent:
        print(f"WARNING: identical code got different Kaggle labels in {len(inconsistent)} groups: {inconsistent[:5]}")
    if mismatches:
        print(f"ERROR: {len(mismatches)} labels could not be reproduced locally:")
        for m in mismatches[:10]:
            print("  ", m)
        sys.exit(1)
    print("all labels reproduced locally")


if __name__ == "__main__":
    main()
