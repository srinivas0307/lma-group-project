"""Stage 1 intent verification: run the intent verifier on every pool candidate.

    .venv/bin/python eval/run_intent_oracle.py                         # oracle (gold) contracts, pool v4 -> results/stage1/oracle
    # Baseline 3 / fine-tuned parsers: contracts from a parser run (eval/parse_contracts.py)
    .venv/bin/python eval/run_intent_oracle.py --parses kaggle_outputs/parsing/7b_p1-v3/parses.jsonl \
        --system intent_prompted_7b_p1v3 --out results/stage1/baseline3      # (eval/score_parsers.py does this for all runs)

With --parses, an output that isn't a valid contract (no JSON, broken JSON, outside vocabulary v1.0) gives
UNCERTAIN / invalid_contract (ESCALATE) for every candidate of that task. The parser runs once per task and its
contract is reused for every candidate (README §4.2).

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
from igt.paths import rel  # noqa: E402
from igt.agents.intent_parser import extract_contract  # noqa: E402
from igt.types import Verdict, execution_failure, uncertain_failure, verdict_record  # noqa: E402
from igt.verifier import verify  # noqa: E402

DIRECTIONS = {"": ("must_change", "must_preserve"), "_change_only": ("must_change",),
              "_preserve_only": ("must_preserve",)}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def gold_contracts(root):
    """task id -> (data_file, gold_contract). Only these two fields are read from the task files."""
    out = {}
    for f in sorted((root / "benchmark" / "tasks").glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        out[t["id"]] = (t["data_file"], t["gold_contract"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "benchmark" / "pool" / "pool_v4.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "stage1" / "oracle")
    ap.add_argument("--parses", type=Path, help="parser output (parses.jsonl); default: gold contracts")
    ap.add_argument("--system", default=None, help="system name (default intent_oracle, or required with --parses)")
    args = ap.parse_args()
    if args.parses and not args.system:
        sys.exit("--system is required with --parses (e.g. intent_prompted_7b)")
    base = args.system or "intent_oracle"
    SYSTEMS = {base + suffix: dirs for suffix, dirs in DIRECTIONS.items()}
    pool = load_jsonl(args.pool)
    tasks = gold_contracts(ROOT)
    parsed = None
    if args.parses:
        parsed = {}
        for r in load_jsonl(args.parses):
            ex = extract_contract(r["raw_output"])
            parsed[r["task_id"]] = ex
        missing = sorted({c["task_id"] for c in pool} - set(parsed))
        if missing:
            sys.exit(f"{args.parses}: no parse for {len(missing)} pool tasks (e.g. {missing[:3]})")
        tasks = {tid: (df, parsed[tid]["contract"] if parsed[tid]["status"] == "ok" else parsed[tid])
                 for tid, (df, _) in tasks.items() if tid in parsed}
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
            if parsed is not None and parsed[c["task_id"]]["status"] != "ok":
                ex = parsed[c["task_id"]]
                v, info = Verdict.combine([uncertain_failure("invalid_contract", f"{ex['status']}: {ex['error']}")]), {}
            else:
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
    (args.out / "run_info.json").write_text(json.dumps(dict(pool=rel(args.pool), n_candidates=len(pool),
                                                            contracts=rel(args.parses) if args.parses else "gold",
                                                            stats={s: dict(v) for s, v in stats.items()}), indent=2))
    for s in SYSTEMS:
        print(s, dict(stats[s]))
    print(f"verdicts in {args.out}")


if __name__ == "__main__":
    main()
