"""Contract metrics for a parser (README Phase 6, guide §5.8): compare predicted contracts with gold.

    .venv/bin/python eval/contract_metrics.py --parses kaggle_outputs/parsing/7b_p1-v3/parses.jsonl

- invalid-output rate: outputs with no JSON, broken JSON, or a contract outside vocabulary v1.0
- per section (must_change, must_preserve): micro precision / recall / F1 over canonical predicate keys
  (an invalid output counts as an empty prediction)
- exact-match accuracy: share of tasks whose keys equal the gold keys in both sections
Reported overall, per split and per transformation type. Writes --out (default <parses dir>/contract_metrics.json;
eval/score_parsers.py puts it in the system's results folder).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.paths import rel  # noqa: E402
from igt.agents.intent_parser import extract_contract  # noqa: E402
from igt.contracts.normalize import contract_keys  # noqa: E402

SECTIONS = ("must_change", "must_preserve")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return dict(precision=None if p is None else round(p, 3), recall=None if r is None else round(r, 3),
                f1=None if f is None else round(f, 3), tp=tp, fp=fp, fn=fn)


def score(rows, gold, columns):
    """rows: parse records. Returns per-task details and aggregate metrics."""
    per_task = []
    for r in rows:
        g = contract_keys(gold[r["task_id"]], columns[r["task_id"]])
        ex = extract_contract(r["raw_output"])
        pred = contract_keys(ex["contract"], columns[r["task_id"]]) if ex["status"] == "ok" else \
            {s: set() for s in SECTIONS}
        d = dict(task_id=r["task_id"], split=r["split"], transformation_type=r["transformation_type"],
                 status=ex["status"], exact=all(pred[s] == g[s] for s in SECTIONS))
        for s in SECTIONS:
            d[s] = dict(tp=len(pred[s] & g[s]), fp=len(pred[s] - g[s]), fn=len(g[s] - pred[s]),
                        missing=sorted(g[s] - pred[s])[:5], extra=sorted(pred[s] - g[s])[:5])
        per_task.append(d)
    return per_task


def aggregate(per_task):
    n = len(per_task)
    out = dict(n_tasks=n, invalid_rate=round(sum(d["status"] != "ok" for d in per_task) / n, 3) if n else None,
               exact_match=round(sum(d["exact"] for d in per_task) / n, 3) if n else None)
    for s in SECTIONS:
        out[s] = prf(*(sum(d[s][k] for d in per_task) for k in ("tp", "fp", "fn")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parses", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = load_jsonl(args.parses)
    gold, columns, cols_cache = {}, {}, {}
    for f in sorted((ROOT / "benchmark" / "tasks").glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        gold[t["id"]] = t["gold_contract"]
        if t["data_file"] not in cols_cache:
            cols_cache[t["data_file"]] = list(pd.read_csv(ROOT / t["data_file"], nrows=1).columns)
        columns[t["id"]] = cols_cache[t["data_file"]]
    per_task = score(rows, gold, columns)
    report = dict(parses=rel(args.parses), overall=aggregate(per_task),
                  by_split={s: aggregate([d for d in per_task if d["split"] == s])
                            for s in sorted({d["split"] for d in per_task})})
    by_type = defaultdict(list)
    for d in per_task:
        if d["split"] == "test":
            by_type[d["transformation_type"]].append(d)
    report["test_by_type"] = {k: aggregate(v) for k, v in sorted(by_type.items())}
    report["per_task"] = per_task
    out = args.out or args.parses.parent / "contract_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    f = lambda m: "  -  " if m is None else f"{m:.3f}"
    print(f"{'scope':<24}{'n':>4}{'invalid':>9}{'exact':>7}{'chg P':>7}{'chg R':>7}{'chg F1':>7}{'prs P':>7}{'prs R':>7}{'prs F1':>7}")
    scopes = [("overall", report["overall"])] + [(f"split={k}", v) for k, v in report["by_split"].items()] + \
        [(f"test {k}", v) for k, v in report["test_by_type"].items()]
    for name, m in scopes:
        c, p = m["must_change"], m["must_preserve"]
        print(f"{name:<24}{m['n_tasks']:>4}{f(m['invalid_rate']):>9}{f(m['exact_match']):>7}"
              f"{f(c['precision']):>7}{f(c['recall']):>7}{f(c['f1']):>7}{f(p['precision']):>7}{f(p['recall']):>7}{f(p['f1']):>7}")
    print(f"written {out}")


if __name__ == "__main__":
    main()
