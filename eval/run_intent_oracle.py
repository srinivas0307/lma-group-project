"""Stage 1 with oracle (gold) contracts: run the intent verifier on every pool candidate.

    .venv/bin/python eval/run_intent_oracle.py                         # pool v2
    .venv/bin/python eval/run_intent_oracle.py --pool eval/pool_v1/pool_v1.jsonl --out results/intent_oracle_v1

Writes three systems (README G4 ablations) in the shared verdict format (docs/verdict_format.md):
    verdicts_intent_oracle.jsonl               full contract
    verdicts_intent_oracle_change_only.jsonl   must_change only
    verdicts_intent_oracle_preserve_only.jsonl must_preserve only
plus run_info.json (alignment methods, uncertain counts).

The verifier only reads: the input CSV, the candidate's output pickle, and the task's gold_contract.
It never reads reference_code, reference outputs or is_correct.
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.types import Verdict, execution_failure, verdict_record  # noqa: E402
from igt.verifier import verify  # noqa: E402

SYSTEMS = {"intent_oracle": ("must_change", "must_preserve"),
           "intent_oracle_change_only": ("must_change",),
           "intent_oracle_preserve_only": ("must_preserve",)}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def gold_contracts(root):
    """task id -> (data_file, gold_contract). Only these two fields are read from the task files."""
    out = {}
    for f in sorted((root / "eval" / "tasks").glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        out[t["id"]] = (t["data_file"], t["gold_contract"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "eval" / "pool_v2" / "pool_v2.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "intent_oracle_v2")
    args = ap.parse_args()
    pool = load_jsonl(args.pool)
    tasks = gold_contracts(ROOT)
    data = {}
    records = {s: [] for s in SYSTEMS}
    stats = {s: Counter() for s in SYSTEMS}
    for c in pool:
        data_file, contract = tasks[c["task_id"]]
        if not c["ran_ok"]:
            v = Verdict.combine([execution_failure(c["status"], c.get("error"))])
            for s in SYSTEMS:
                records[s].append(verdict_record(s, "intent", c, v))
                stats[s]["execution"] += 1
            continue
        if data_file not in data:
            data[data_file] = pd.read_csv(ROOT / data_file)
        with open(args.pool.parent / c["output_file"], "rb") as fh:
            after = pickle.load(fh)
        for s, dirs in SYSTEMS.items():
            v, info = verify(contract, data[data_file], after, dirs)
            rec = verdict_record(s, "intent", c, v)
            rec["meta"] = dict(alignment=info.get("alignment"), n_predicates=info.get("n_predicates"))
            records[s].append(rec)
            stats[s][f"alignment_{info.get('alignment')}"] += 1
            stats[s][v.status] += 1
            for f in v.failures:
                if f["kind"] == "uncertain":
                    stats[s][f"uncertain_{f['reason']}"] += 1
    args.out.mkdir(parents=True, exist_ok=True)
    for s, rows in records.items():
        with open(args.out / f"verdicts_{s}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    (args.out / "run_info.json").write_text(json.dumps(dict(pool=str(args.pool), n_candidates=len(pool),
                                                            stats={s: dict(v) for s, v in stats.items()}), indent=2))
    for s in SYSTEMS:
        print(s, dict(stats[s]))
    print(f"verdicts in {args.out}")


if __name__ == "__main__":
    main()
