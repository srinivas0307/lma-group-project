"""Build a frozen candidate pool from one or more labelled generation folders (CPU only).

    # pool v1 (already frozen): Baseline 1 Kaggle outputs
    .venv/bin/python eval/build_pool.py --sources notebooks/baseline1_outputs --version v1
    # pool v2: v1 sources first (so v1 candidate ids stay the same), then the new generator
    .venv/bin/python eval/build_pool.py --sources notebooks/baseline1_outputs outputs/gen_qwen15b --version v2

Each source folder needs candidates.jsonl (from the Baseline 1 notebook or eval/label_candidates.py).

Outputs (in eval/pool_<version>/):
    pool_<version>.jsonl        one row per unique (task, normalized code); what every system judges
    outputs/<candidate_id>.pkl  the candidate's output DataFrame (candidates that ran)
    verdicts_baseline1.jsonl    Baseline 1 verdicts (verifier "none")
    manifest.json               counts, sources, settings and sha256 of every file

Every candidate is re-executed with the project sandbox (igt.sandbox.safe_run) and re-labelled with
eval.equivalence.outputs_equal. Any label that differs from the source label stops the build.
"""
import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.equivalence import outputs_equal  # noqa: E402
from igt.agents.code_generator import load_tasks  # noqa: E402
from igt.sandbox import safe_run  # noqa: E402

PINNED = {"pandas": "2.3.3"}
CRASH = {"error", "bad_return", "timeout", "killed", "memory"}


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


def load_sources(sources):
    gens = []
    for si, src in enumerate(sources):
        rows = load_jsonl(src / "candidates.jsonl")
        for field in ["model", "prompt_hash"]:
            vals = {g[field] for g in rows}
            if len(vals) != 1:
                sys.exit(f"{src}: candidates.jsonl mixes several values of {field}: {vals}")
        for g in rows:
            g["_source"] = si
        gens += rows
    hashes = {g["prompt_hash"] for g in gens}
    if len(hashes) != 1:
        sys.exit(f"sources use different prompts ({hashes}); a pool must use one prompt version")
    keys = Counter((g["model"], g["gen_id"]) for g in gens)
    if any(n > 1 for n in keys.values()):
        sys.exit(f"duplicate (model, gen_id): {[k for k, n in keys.items() if n > 1][:5]}")
    return gens


def order_key(g):
    return (g["_source"], g["mode"] != "greedy", g["sample_idx"])


def group_candidates(gens):
    """One pool entry per unique (task, normalized code). Unparsed outputs are keyed by raw text."""
    groups = defaultdict(list)
    for g in gens:
        key = g.get("code_hash") or "raw:" + hashlib.sha256(g["raw_output"].encode()).hexdigest()[:16]
        groups[(g["task_id"], key)].append(g)
    pool = []
    for task_id in sorted({k[0] for k in groups}):
        keys = sorted((k for k in groups if k[0] == task_id), key=lambda k: min(map(order_key, groups[k])))
        for n, k in enumerate(keys):
            members = sorted(groups[k], key=order_key)
            first = members[0]
            labels = {(m["ran_ok"], m["correct"]) for m in members}
            pool.append(dict(
                candidate_id=f"{task_id}__c{n:02d}", task_id=task_id, split=first["split"],
                dataset=first["dataset"], transformation_type=first["transformation_type"],
                difficulty=first.get("difficulty"), generator_model=first["model"],
                generator_models=sorted({m["model"] for m in members}),
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


def verdict_b1(c):
    """Baseline 1 = verifier 'none': PASS whenever the code ran; anything else is REJECT (repair off)."""
    if c["ran_ok"]:
        v, outcome = dict(status="PASS", failures=[], evidence_rows=[]), "ACCEPT"
    else:
        v = dict(status="VIOLATION", failures=[dict(kind="execution", status=c["status"], detail=c.get("error"))],
                 evidence_rows=[])
        outcome = "REJECT"
    return dict(system="baseline1", verifier="none", candidate_id=c["candidate_id"], task_id=c["task_id"],
                split=c["split"], verdict=v, outcome=outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, nargs="+", help="labelled generation folders, in priority order")
    ap.add_argument("--b1", type=Path, help="(old name) single source folder")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, help="default: eval/pool_<version>")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--cpu", type=int, default=30)
    ap.add_argument("--mem", type=float, default=4)
    ap.add_argument("--force", action="store_true", help="overwrite an existing pool")
    args = ap.parse_args()
    sources = [p.resolve() for p in (args.sources or ([args.b1] if args.b1 else []))]
    if not sources:
        sys.exit("give --sources")
    out = (args.out or ROOT / "eval" / f"pool_{args.version}").resolve()
    pool_file = out / f"pool_{args.version}.jsonl"
    if pd.__version__ != PINNED["pandas"]:
        sys.exit(f"pandas {PINNED['pandas']} required, got {pd.__version__}. Use .venv/bin/python.")
    if pool_file.exists() and not args.force:
        sys.exit(f"{pool_file} already exists; the pool is frozen. Use --force only if you mean it.")

    gens = load_sources(sources)
    tasks = {t["id"]: t for t in load_tasks(args.root)}
    data = {f: pd.read_csv(args.root / f) for f in {t["data_file"] for t in tasks.values()}}
    limits = dict(timeout_s=args.timeout, cpu_s=args.cpu, mem_gb=args.mem)
    refs = {}
    for tid in sorted({g["task_id"] for g in gens}):
        r = safe_run(tasks[tid]["reference_code"], data[tasks[tid]["data_file"]], **limits)
        if not r.ran_ok:
            sys.exit(f"reference for {tid} failed: {r.status} {r.error}")
        refs[tid] = r.df_out

    pool = group_candidates(gens)
    out_dir = out / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    def rerun(c):
        if not c["parse_ok"] or c["blocked"]:
            return dict(status="parse_error" if not c["parse_ok"] else "blocked", ran_ok=False, correct=False)
        r = safe_run(c["code"], data[tasks[c["task_id"]]["data_file"]], c["func_name"], **limits)
        if not r.ran_ok:
            return dict(status=r.status, ran_ok=False, correct=False, error=r.error)
        ok, reason = outputs_equal(refs[c["task_id"]], r.df_out)
        blob = pickle.dumps(r.df_out)
        (out_dir / f"{c['candidate_id']}.pkl").write_bytes(blob)
        return dict(status="ok", ran_ok=True, correct=bool(ok), reason=reason,
                    output_sha256=hashlib.sha256(blob).hexdigest())

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(rerun, pool))

    mismatches = []
    for c, r in zip(pool, results):
        c.update(status=r["status"], ran_ok=bool(r["ran_ok"]), is_correct=bool(r["correct"]),
                 reason=r.get("reason"), error=r.get("error"),
                 output_file=f"outputs/{c['candidate_id']}.pkl" if r["ran_ok"] else None,
                 output_sha256=r.get("output_sha256"))
        if (c["ran_ok"], c["is_correct"]) != (c["kaggle_ran_ok"], c["kaggle_correct"]):
            mismatches.append(dict(candidate_id=c["candidate_id"], source=(c["kaggle_ran_ok"], c["kaggle_correct"]),
                                   local=(c["ran_ok"], c["is_correct"]), local_error=c["error"]))
    inconsistent = [c["candidate_id"] for c in pool if not c["consistent_within_group"]]

    write_jsonl(pool_file, pool)
    write_jsonl(out / "verdicts_baseline1.jsonl", [verdict_b1(c) for c in pool])

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
            runnable_wrong_by_generator=dict(sorted(Counter(m for c in S if c["ran_ok"] and not c["is_correct"]
                                                            for m in c["generator_models"]).items())),
        )
    envs = []
    for s in sources:
        e = s / "env.json"
        envs.append(dict(path=str(s), env=json.loads(e.read_text()) if e.exists() else None,
                         sha256_candidates_jsonl=sha256_file(s / "candidates.jsonl")))
    manifest = dict(
        pool_version=args.version, sources=envs, local_versions=dict(pandas=pd.__version__),
        n_generations=len(gens), n_candidates=len(pool), summary=summary,
        label_mismatches=mismatches, inconsistent_groups=inconsistent,
        sha256={p.relative_to(out).as_posix(): sha256_file(p)
                for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"{len(gens)} generations from {len(sources)} source(s) -> {len(pool)} unique candidates -> {out}")
    for split, s in summary.items():
        print(f"  {split}: {s}")
    if inconsistent:
        print(f"WARNING: identical code got different source labels in {len(inconsistent)} groups: {inconsistent[:5]}")
    if mismatches:
        print(f"ERROR: {len(mismatches)} labels could not be reproduced locally:")
        for m in mismatches[:10]:
            print("  ", m)
        sys.exit(1)
    print("all labels reproduced locally")


if __name__ == "__main__":
    main()
