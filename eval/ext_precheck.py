"""Extension #9 (pc1), offline: apply the contract pre-check (src/igt/contracts/precheck.py) to existing parser runs.
No model calls, no GPU. A valid contract with a pre-check issue is treated as invalid (the task is escalated).

    .venv/bin/python eval/ext_precheck.py 7b_p1-v3 3b_p1-v2 1_5b_p1-v3      # folders under kaggle_outputs/parsing/
    .venv/bin/python eval/ext_precheck.py 7b_p1-v3 --no-score                # only write the derived parses

Per input folder, writes results/stage1/extensions/<system>_pc1/:
    parses.jsonl     the same records; flagged ones get raw_output = a note with no JSON (so every scorer reads them as
                     invalid), plus raw_output_original, precheck_version, precheck_issues
    system.txt       <system>_pc1
    precheck_report.json   per check: how many contracts it flagged, and how many of those differ from the gold
                           contract (contract_metrics `exact` is False) = useful flags vs false alarms
then scores them with eval/ext_score.py (verdicts + contract metrics, summary in results/stage1/extensions/summary/).
The input parse folders are only read.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ext_common import ext_dir, load_jsonl, parse_folder, tasks_and_data, write_jsonl  # noqa: E402
from igt.agents.intent_parser import extract_contract  # noqa: E402
from igt.contracts.precheck import PRECHECK_VERSION, issues_text, precheck  # noqa: E402
from igt.paths import rel  # noqa: E402


def flagged_note(issues):
    # No braces, so no scorer finds a JSON object in it.
    return ("[pre-check pc1 flagged this contract; treated as invalid]\n" + issues_text(issues)).replace("{", "(").replace("}", ")")


def apply_precheck(rows, tasks, data):
    out = []
    for r in rows:
        r = dict(r)
        ex = extract_contract(r["raw_output"])
        issues = []
        if ex["status"] == "ok":
            t = tasks[r["task_id"]]
            issues = precheck(ex["contract"], data[t["data_file"]])
        r.update(precheck_version=PRECHECK_VERSION, precheck_issues=issues, raw_output_original=r["raw_output"])
        if issues:
            r.update(raw_output=flagged_note(issues), parse_status="precheck_flagged",
                     parse_error="; ".join(i["check"] for i in issues))
        out.append(r)
    return out


def report(rows, tasks):
    from eval.contract_metrics import score
    import pandas as pd
    cols = {}
    for tid, t in tasks.items():
        cols[tid] = list(pd.read_csv(Path(__file__).resolve().parent.parent / t["data_file"], nrows=1).columns)
    orig = [dict(r, raw_output=r["raw_output_original"]) for r in rows]
    exact = {d["task_id"]: d["exact"] for d in score(orig, {k: t["gold_contract"] for k, t in tasks.items()}, cols)}
    by = defaultdict(lambda: Counter())
    flagged = [r for r in rows if r["precheck_issues"]]
    for r in flagged:
        for c in sorted({i["check"] for i in r["precheck_issues"]}):
            by[c]["flagged"] += 1
            by[c]["wrong_vs_gold" if not exact[r["task_id"]] else "matches_gold"] += 1
    by_split = Counter(r["split"] for r in flagged)
    return dict(precheck_version=PRECHECK_VERSION, n_tasks=len(rows),
                valid_before=sum(extract_contract(r["raw_output_original"])["status"] == "ok" for r in rows),
                flagged=len(flagged), flagged_by_split=dict(by_split),
                flagged_wrong_vs_gold=sum(not exact[r["task_id"]] for r in flagged),
                flagged_matches_gold=sum(exact[r["task_id"]] for r in flagged),
                by_check={k: dict(v) for k, v in sorted(by.items())},
                tasks=[dict(task_id=r["task_id"], split=r["split"], exact_vs_gold=exact[r["task_id"]],
                            issues=r["precheck_issues"]) for r in flagged])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="parse folders (name under kaggle_outputs/parsing/ or a path)")
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()
    tasks, data = tasks_and_data()
    outs = []
    for f in args.folders:
        d, base = parse_folder(f)
        system = f"{base}_pc1"
        out = ext_dir(system)
        rows = apply_precheck(load_jsonl(d / "parses.jsonl"), tasks, data)
        write_jsonl(out / "parses.jsonl", rows)
        (out / "system.txt").write_text(system + "\n")
        rep = dict(source=rel(d), base_system=base, **report(rows, tasks))
        (out / "precheck_report.json").write_text(json.dumps(rep, indent=2, default=str))
        print(f"{system}: {rep['flagged']} of {rep['valid_before']} valid contracts flagged "
              f"({rep['flagged_wrong_vs_gold']} differ from gold, {rep['flagged_matches_gold']} match gold); "
              f"by check {json.dumps(rep['by_check'])}")
        outs.append(out)
    if not args.no_score:
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).with_name("ext_score.py")), *map(str, outs)], check=True)


if __name__ == "__main__":
    main()
