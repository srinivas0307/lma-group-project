"""Run Baseline 2 (conventional checks) over the frozen pool: measure, tune on dev, judge everything.

    .venv/bin/python eval/baseline2.py                     # uses configs/baseline2_v4.yaml

Outputs in results/stage1/baseline2/ (out_dir in the config):
    measurements.jsonl          raw check measurements per runnable candidate
    thresholds.json             tuned thresholds + the tuning trace (dev only)
    verdicts_baseline2.jsonl    tuned verdicts, same format as Baseline 1 (this is "Baseline 2")
    verdicts_baseline2_default.jsonl  untuned default thresholds (sensitivity check)
"""
import argparse, json, pickle, sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.baselines.conventional_checks import judge, measure  # noqa: E402

PINNED = {"pandas": "2.3.3"}


def load_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")


def task_data_files(root):
    out = {}
    for f in sorted((root / "benchmark" / "tasks").glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        out[t["id"]] = t["data_file"]
    return out


def compute_measurements(pool, pool_dir, root):
    files = task_data_files(root)
    inputs = {}
    out = {}
    for c in pool:
        if not c["ran_ok"]:
            continue
        f = files[c["task_id"]]
        if f not in inputs:
            inputs[f] = pd.read_csv(root / f)
        with open(pool_dir / c["output_file"], "rb") as fh:
            after = pickle.load(fh)
        out[c["candidate_id"]] = measure(inputs[f], after)
    return out


def verdict(system, c, m, thresholds):
    if not c["ran_ok"]:
        v = dict(status="VIOLATION", failures=[dict(kind="execution", status=c["status"], detail=c.get("error"))],
                 evidence_rows=[])
    else:
        v = judge(m[c["candidate_id"]], thresholds)
    return dict(system=system, verifier="conventional", candidate_id=c["candidate_id"], task_id=c["task_id"],
                split=c["split"], verdict=v, outcome="ACCEPT" if v["status"] == "PASS" else "REJECT")


def score(cands, m, thresholds):
    """Dev objective over runnable candidates: (recall on wrong, false-rejection rate on correct)."""
    wrong = rej_w = corr = rej_c = 0
    for c in cands:
        rejected = judge(m[c["candidate_id"]], thresholds)["status"] != "PASS"
        if c["is_correct"]:
            corr += 1
            rej_c += rejected
        else:
            wrong += 1
            rej_w += rejected
    return (rej_w / wrong if wrong else 0.0), (rej_c / corr if corr else 0.0)


def tune(cands, m, grid, cap, max_passes, tie_break="strict"):
    """Coordinate ascent: maximise recall s.t. FRR <= cap; ties -> lower FRR, then by tie_break.

    tie_break="strict": among equally good values keep the strictest, so a check that costs nothing
    on dev stays on (strong-baseline choice). "lenient": keep the most lenient (switches such checks off).
    """
    sign = 1 if tie_break == "strict" else -1
    th = {k: v[0] for k, v in grid.items()}                     # start fully lenient
    trace = []
    best = score(cands, m, th)
    for p in range(max_passes):
        changed = False
        for k, values in grid.items():
            options = []
            for i, v in enumerate(values):
                rec, frr = score(cands, m, {**th, k: v})
                if frr <= cap + 1e-12:
                    options.append((rec, -frr, sign * i, v))
            if not options:
                continue
            rec, nfrr, _, v = max(options)
            if v != th[k]:
                trace.append(dict(pass_=p, param=k, old=th[k], new=v, recall=round(rec, 3), frr=round(-nfrr, 3)))
                th[k] = v
                changed = True
                best = (rec, -nfrr)
        if not changed:
            break
    return th, dict(recall=round(best[0], 3), frr=round(best[1], 3)), trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline2_v4.yaml")
    args = ap.parse_args()
    if pd.__version__ != PINNED["pandas"]:
        sys.exit(f"pandas {PINNED['pandas']} required (pickled outputs), got {pd.__version__}. Use .venv/bin/python.")
    cfg = yaml.safe_load(args.config.read_text())
    pool_path = ROOT / cfg["pool"]
    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = load_jsonl(pool_path)

    m = compute_measurements(pool, pool_path.parent, ROOT)
    write_jsonl(out_dir / "measurements.jsonl", [dict(candidate_id=k, **v) for k, v in m.items()])

    tcfg = cfg["tuning"]
    dev = [c for c in pool if c["split"] == tcfg["split"] and c["ran_ok"]]
    tie = tcfg.get("tie_break", "lenient")
    other = "lenient" if tie == "strict" else "strict"
    th, dev_score, trace = tune(dev, m, cfg["grid"], tcfg["max_false_rejection_rate"], tcfg["max_passes"], tie)
    th_alt, alt_score, alt_trace = tune(dev, m, cfg["grid"], tcfg["max_false_rejection_rate"],
                                       tcfg["max_passes"], other)
    default = cfg["default_thresholds"]
    rnd = lambda s: dict(zip(["recall", "frr"], (round(x, 3) for x in s)))
    (out_dir / "thresholds.json").write_text(json.dumps(dict(
        tuning_split=tcfg["split"], n_dev_runnable=len(dev), max_false_rejection_rate=tcfg["max_false_rejection_rate"],
        tuned=dict(tie_break=tie, thresholds=th, dev_score=dev_score, trace=trace),
        sensitivity_tie_break=dict(tie_break=other, thresholds=th_alt, dev_score=alt_score, trace=alt_trace),
        default=dict(thresholds=default, dev_score=rnd(score(dev, m, default)))), indent=2, default=str))

    write_jsonl(out_dir / "verdicts_baseline2.jsonl", [verdict("baseline2", c, m, th) for c in pool])
    write_jsonl(out_dir / f"verdicts_baseline2_{other}.jsonl",
                [verdict(f"baseline2_{other}", c, m, th_alt) for c in pool])
    write_jsonl(out_dir / "verdicts_baseline2_default.jsonl",
                [verdict("baseline2_default", c, m, default) for c in pool])

    print(f"measured {len(m)} runnable candidates; tuned on {len(dev)} dev candidates")
    print(f"tuned thresholds (tie_break={tie}):", json.dumps(th))
    print("dev score:", dev_score, f"| {other} tie-break dev score:", alt_score)
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
