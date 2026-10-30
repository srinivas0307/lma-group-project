"""Check fine-tuning examples before they are used (docs/finetune_data_guide.md §6).

    .venv/bin/python finetune/data_gen/check_examples.py finetune/data/raw/*.jsonl --out finetune/data/checked.jsonl

Input: JSONL, one example per line:
    {"id": "wine_filt_0001", "source_dataset": "wine", "data_file": "finetune/datasets/wine.csv",
     "transformation_type": "filtering", "pattern": "filter_numeric_threshold", "style": "casual",
     "request": "...", "contract": {...vocabulary v1.0...}, "reference_code": "def transform(df): ..."}

Each example must pass every check, otherwise it is dropped and the reason is reported:
  1. data      the data file is not an evaluation dataset (by path and by column set)
  2. schema    the contract is valid under vocabulary v1.0 and only names columns of its data
  3. runs      the reference code passes the AST guard and runs in the sandbox
  4. passes    the reference output PASSES the contract (intent verifier)
  5. nonvacuous returning the input unchanged does NOT pass
  6. leakage   the request is not too close to any evaluation request (word 5-gram overlap, character similarity)
  7. unique    no duplicate request text in the training data
Writes the passing examples to --out and a report next to it.
"""
import argparse
import difflib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from igt.contracts import ContractError, load_contract, predicate_columns  # noqa: E402
from igt.sandbox import safe_run  # noqa: E402
from igt.verifier import verify  # noqa: E402

NGRAM = 5
MAX_SHARED_NGRAMS = 0          # any shared 5-gram with an evaluation request is a leak
MAX_CHAR_SIMILARITY = 0.75     # difflib ratio against every evaluation request
REQUIRED = ("id", "source_dataset", "data_file", "transformation_type", "request", "contract", "reference_code")
TYPES = ("filtering", "deduplication", "missing_values", "type_conversion", "column_transformation")


def words(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def ngrams(text, n=NGRAM):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_eval(root):
    tasks = [yaml.safe_load(f.read_text()) for f in sorted((root / "benchmark" / "tasks").glob("*/*.yaml"))]
    requests = [" ".join(str(t["request"]).split()) for t in tasks]
    files = {t["data_file"] for t in tasks}
    colsets = {frozenset(pd.read_csv(root / f, nrows=1).columns) for f in files}
    names = {Path(f).name.lower() for f in files}
    return requests, files, colsets, names


class Checker:
    def __init__(self, root, limits):
        self.root = root
        self.limits = limits
        self.eval_requests, self.eval_files, self.eval_colsets, self.eval_names = load_eval(root)
        self.eval_ngrams = set().union(*(ngrams(r) for r in self.eval_requests))
        self.data = {}

    def df(self, data_file):
        if data_file not in self.data:
            self.data[data_file] = pd.read_csv(self.root / data_file)
        return self.data[data_file]

    def check(self, ex):
        miss = [k for k in REQUIRED if k not in ex]
        if miss:
            return "format", f"missing fields {miss}"
        if ex["transformation_type"] not in TYPES:
            return "format", f"transformation_type must be one of {TYPES}"
        f = ex["data_file"]
        if f in self.eval_files or Path(f).name.lower() in self.eval_names:
            return "data", f"{f} is an evaluation dataset"
        if not (self.root / f).exists():
            return "data", f"{f} not found"
        before = self.df(f)
        if frozenset(before.columns) in self.eval_colsets:
            return "data", "same columns as an evaluation dataset"
        try:
            c = load_contract(ex["contract"])
        except ContractError as e:
            return "schema", str(e)
        cols = list(before.columns)
        unknown = sorted({x for _, _, p in c.predicates() for x in predicate_columns(p, cols) if x not in cols})
        if unknown:
            return "schema", f"unknown columns {unknown}"
        if not c.must_change:
            return "schema", "must_change is empty"
        r = safe_run(ex["reference_code"], before, **self.limits)
        if not r.ran_ok:
            return "runs", f"{r.status}: {r.error}"
        v, _ = verify(c, before, r.df_out)
        if v.status != "PASS":
            return "passes", v.explain()[:400]
        v0, _ = verify(c, before, before.copy())
        if v0.status == "PASS":
            return "nonvacuous", "the unchanged input passes the contract"
        shared = ngrams(ex["request"]) & self.eval_ngrams
        if len(shared) > MAX_SHARED_NGRAMS:
            return "leakage", f"shares {len(shared)} word {NGRAM}-grams with evaluation requests, e.g. {' '.join(next(iter(shared)))!r}"
        req = " ".join(ex["request"].split()).lower()
        best = max(difflib.SequenceMatcher(None, req, e.lower()).ratio() for e in self.eval_requests)
        if best > MAX_CHAR_SIMILARITY:
            return "leakage", f"character similarity {best:.2f} with an evaluation request"
        return "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    examples = [json.loads(l) for p in args.inputs for l in p.read_text().splitlines() if l.strip()]
    ids = Counter(e.get("id") for e in examples)
    reqs = Counter(" ".join(e.get("request", "").split()).lower() for e in examples)
    chk = Checker(ROOT, dict(timeout_s=30, cpu_s=15, mem_gb=2))
    for f in {e.get("data_file") for e in examples if e.get("data_file")}:
        if (ROOT / f).exists():
            chk.df(f)                      # load data before threading
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(chk.check, examples))
    kept, report = [], []
    for e, (status, why) in zip(examples, results):
        if status == "ok" and ids[e["id"]] > 1:
            status, why = "unique", "duplicate id"
        if status == "ok" and reqs[" ".join(e["request"].split()).lower()] > 1:
            status, why = "unique", "duplicate request text"
        report.append(dict(id=e.get("id"), status=status, reason=why))
        if status == "ok":
            kept.append(e)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for e in kept:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    rep_path = args.out.with_suffix(".report.json")
    summary = dict(
        n_input=len(examples), n_kept=len(kept), by_status=dict(Counter(r["status"] for r in report)),
        kept_by_type=dict(Counter(e["transformation_type"] for e in kept)),
        kept_by_dataset=dict(Counter(e["source_dataset"] for e in kept)),
        kept_by_predicate=dict(Counter(p["type"] for e in kept for d in ("must_change", "must_preserve")
                                       for p in e["contract"].get(d) or [])),
        kept_by_style=dict(Counter(e.get("style", "?") for e in kept)),
    )
    rep_path.write_text(json.dumps(dict(summary=summary, dropped=[r for r in report if r["status"] != "ok"]), indent=2))
    print(json.dumps(summary, indent=2))
    for r in [r for r in report if r["status"] != "ok"][:10]:
        print(f"DROPPED {r['id']}: {r['status']}: {r['reason'][:200]}")
    print(f"kept {len(kept)}/{len(examples)} -> {args.out}; report {rep_path}")


if __name__ == "__main__":
    main()
