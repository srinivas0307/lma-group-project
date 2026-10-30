"""Label raw generations (CPU, pinned pandas): extract code, AST guard, sandbox run, compare with the reference.

    .venv/bin/python eval/label_candidates.py --gen kaggle_outputs/generation/1_5b_adult_airbnb

Reads  <gen>/generations.jsonl (from eval/generate.py or the Baseline 1 notebook)
Writes <gen>/candidates.jsonl  (same fields as the Baseline 1 notebook) and <gen>/label_summary.json
"""
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.equivalence import norm, outputs_equal  # noqa: E402
from igt.agents.code_extract import extract_code  # noqa: E402
from igt.agents.code_generator import load_tasks  # noqa: E402
from igt.sandbox import check, safe_run  # noqa: E402

PINNED = {"pandas": "2.3.3"}


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def index_kept(before, out):
    """Informational: is the output index still usable as the row id?"""
    try:
        if not out.index.is_unique or not out.index.isin(before.index).all():
            return False
        common = [c for c in before.columns if c in out.columns]
        if not common or len(out) == 0:
            return True
        bv = before.loc[out.index, common].astype(object).values
        ov = out[common].astype(object).values
        agree = sum(2 * sum(norm(x) == norm(y) for x, y in zip(b, o)) >= len(common) for b, o in zip(bv, ov))
        return agree / len(out) >= 0.95
    except Exception:
        return False


class Labeller:
    def __init__(self, root, limits):
        self.root = root
        self.limits = limits
        self.tasks = {t["id"]: t for t in load_tasks(root)}
        self.data, self.refs = {}, {}

    def df(self, task):
        f = task["data_file"]
        if f not in self.data:
            self.data[f] = pd.read_csv(self.root / f)
        return self.data[f]

    def ref(self, task):
        if task["id"] not in self.refs:
            r = safe_run(task["reference_code"], self.df(task), **self.limits)
            if not r.ran_ok:
                raise RuntimeError(f"reference for {task['id']} failed: {r.status} {r.error}")
            self.refs[task["id"]] = r.df_out
        return self.refs[task["id"]]

    def label(self, gen):
        task = self.tasks[gen["task_id"]]
        rec = dict(gen)
        rec.update(parse_ok=False, blocked=False, ran_ok=False, correct=False, status=None, error=None, reason=None)
        ex = extract_code(gen["raw_output"])
        rec.update(ex)
        if not ex["parse_ok"]:
            rec["status"] = "parse_error"
            return rec
        g = check(ex["code"])
        if not g.ok:
            rec.update(status="blocked", blocked=True, error=g.reason)
            return rec
        df = self.df(task)
        r = safe_run(ex["code"], df, ex["func_name"], **self.limits)
        rec.update(status=r.status, error=r.error, runtime_s=r.duration_s, rows_before=len(df))
        if r.ran_ok:
            ok, reason = outputs_equal(self.ref(task), r.df_out)
            out = r.df_out
            rec.update(ran_ok=True, correct=bool(ok), reason=reason, rows_after=len(out),
                       cols_added=sorted(map(str, set(out.columns) - set(df.columns))),
                       cols_removed=sorted(map(str, set(df.columns) - set(out.columns))),
                       index_kept=bool(index_kept(df, out)))
        rec["accepted_baseline1"] = bool(rec["ran_ok"])
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True, help="folder with generations.jsonl")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--cpu", type=int, default=30)
    ap.add_argument("--mem", type=float, default=4)
    args = ap.parse_args()
    if pd.__version__ != PINNED["pandas"]:
        sys.exit(f"pandas {PINNED['pandas']} required, got {pd.__version__}. Use .venv/bin/python.")
    gens = load_jsonl(args.gen / "generations.jsonl")
    if not gens:
        sys.exit(f"no generations in {args.gen / 'generations.jsonl'}")
    lab = Labeller(args.root.resolve(), dict(timeout_s=args.timeout, cpu_s=args.cpu, mem_gb=args.mem))
    for tid in sorted({g["task_id"] for g in gens}):       # compute references up front (not in threads)
        lab.ref(lab.tasks[tid])
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(lab.label, gens))
    with open(args.gen / "candidates.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, default=str) + "\n")

    summary = {}
    for split in sorted({r["split"] for r in recs}):
        S = [r for r in recs if r["split"] == split]
        summary[split] = dict(
            generations=len(S), status=dict(Counter(r["status"] for r in S)),
            correct=sum(r["correct"] for r in S), runnable_wrong=sum(r["ran_ok"] and not r["correct"] for r in S),
            unique_runnable_wrong_by_type=dict(sorted(Counter(
                r["transformation_type"] for r in {(r["task_id"], r["code_hash"]): r for r in S
                                                   if r["ran_ok"] and not r["correct"]}.values()).items())))
    (args.gen / "label_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"labelled {len(recs)} generations in {time.time() - t0:.0f}s -> {args.gen / 'candidates.jsonl'}")
    for split, s in summary.items():
        print(f"  {split}: {s}")


if __name__ == "__main__":
    main()
