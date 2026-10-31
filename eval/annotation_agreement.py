"""Agreement between the first (gold) and the second annotator (guide §12.3, OI-8). CPU only.

    .venv/bin/python eval/annotation_agreement.py --check     # validate the filled sheets, print what's left to do
    .venv/bin/python eval/annotation_agreement.py             # score the sheets marked status: done

Sheets: benchmark/annotation/second_pass/{tasks,scenarios}/*.yaml (eval/make_annotation_sheet.py).
Writes benchmark/annotation/agreement.json and agreement.md (numbers + one row per disagreement for the resolution log).

Measures (on sheets with status: done; report them BEFORE fixing anything, guide §12.3):
  1. Predicate-level P / R / F1 per section (gold = reference; same matching as the parser metrics, docs/contract_schema.md
     §10) and exact-match rate. Target: F1 >= 0.85 per section.
  2. Behaviour: does the second contract accept the reference output and reject the unchanged input (no-op)?
  3. Decision agreement on the real LLM candidates of each task (pool v4, runnable ones): accept vs not-accept with the
     gold contract (the oracle verdicts) and with the second contract; % agreement and Cohen's kappa, plus how many
     correct candidates the second contract rejects and how many wrong ones it accepts (answer key = pool labels).
  4. Scenario outcomes: % agreement on expected_outcome. Target >= 90%.
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.contract_metrics import prf  # noqa: E402
from igt.contracts import ContractError, load_contract  # noqa: E402
from igt.contracts.normalize import contract_keys  # noqa: E402
from igt.paths import AGENT_SCENARIOS, POOL, STAGE1, TASKS  # noqa: E402
from igt.sandbox import safe_run  # noqa: E402
from igt.verifier import verify  # noqa: E402

SHEETS = ROOT / "benchmark" / "annotation" / "second_pass"
SECTIONS = ("must_change", "must_preserve")
OUTCOMES = {"ACCEPT", "ESCALATE", "REJECT_OR_ESCALATE"}


def load_sheets(d):
    out = []
    for f in sorted(Path(d).glob("*.yaml")):
        s = yaml.safe_load(f.read_text()) or {}
        s["_file"] = f
        out.append(s)
    return out


def check_task_sheet(s, gold_task):
    """-> list of problems (empty = OK)."""
    probs = []
    if s.get("status") != "done":
        return ["status is not 'done'"]
    if not str(s.get("annotator") or "").strip():
        probs.append("annotator is empty")
    c = s.get("contract") or {}
    if not (c.get("must_change") or c.get("must_preserve")):
        probs.append("contract is empty")
    try:
        load_contract(c)
    except ContractError as e:
        probs.append(f"invalid contract: {str(e)[:300]}")
    if gold_task is None:
        probs.append("unknown task_id")
    return probs


def kappa(pairs):
    """Cohen's kappa for two binary raters; pairs = [(a, b)]."""
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else (1.0 if po == 1 else None)


def outcome(v):
    return "ACCEPT" if v.status == "PASS" else "ESCALATE" if v.status == "UNCERTAIN" else "REJECT"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", type=Path, default=SHEETS)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--oracle-verdicts", type=Path, default=STAGE1 / "oracle" / "verdicts_intent_oracle.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="folder for agreement.json/.md (default: the sheets' parent)")
    ap.add_argument("--check", action="store_true", help="only validate the sheets")
    args = ap.parse_args(argv)
    out_dir = args.out or args.sheets.parent

    gold = {t["id"]: t for t in (yaml.safe_load(f.read_text()) for f in sorted(TASKS.glob("*/*.yaml")))}
    scen_gold = {s["id"]: s for c in ("E", "S") for s in (yaml.safe_load(f.read_text())
                                                          for f in sorted((AGENT_SCENARIOS / c).glob("*.yaml")))}
    tsheets, ssheets = load_sheets(args.sheets / "tasks"), load_sheets(args.sheets / "scenarios")
    problems = {}
    for s in tsheets:
        p = check_task_sheet(s, gold.get(s.get("task_id")))
        if p:
            problems[s["_file"].name] = p
    for s in ssheets:
        if s.get("status") != "done":
            problems[s["_file"].name] = ["status is not 'done'"]
        elif s.get("expected_outcome") not in OUTCOMES:
            problems[s["_file"].name] = [f"expected_outcome must be one of {sorted(OUTCOMES)}"]
    done_t = [s for s in tsheets if s["_file"].name not in problems]
    done_s = [s for s in ssheets if s["_file"].name not in problems]
    print(f"task sheets ready {len(done_t)}/{len(tsheets)} | scenario sheets ready {len(done_s)}/{len(ssheets)}")
    for name, p in sorted(problems.items()):
        if p != ["status is not 'done'"]:
            print(f"  {name}: {'; '.join(p)}")
    if args.check:
        return dict(ready_tasks=len(done_t), ready_scenarios=len(done_s), problems=problems)
    if not done_t and not done_s:
        sys.exit("no sheet has status: done yet")

    pool = [json.loads(l) for l in args.pool.read_text().splitlines() if l.strip()]
    oracle = {}
    for l in args.oracle_verdicts.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            oracle[r["candidate_id"]] = r["outcome"]
    data, per_task, rows_md = {}, [], []
    tot = {s: [0, 0, 0] for s in SECTIONS}
    all_pairs = []
    for s in done_t:
        t = gold[s["task_id"]]
        if t["data_file"] not in data:
            data[t["data_file"]] = pd.read_csv(ROOT / t["data_file"])
        before = data[t["data_file"]]
        cols = list(before.columns)
        g, a = contract_keys(t["gold_contract"], cols), contract_keys(s["contract"], cols)
        d = dict(task_id=t["id"], transformation_type=t["transformation_type"], annotator=s["annotator"],
                 exact=all(g[x] == a[x] for x in SECTIONS))
        for x in SECTIONS:
            tp, fp, fn = len(a[x] & g[x]), len(a[x] - g[x]), len(g[x] - a[x])
            tot[x] = [tot[x][0] + tp, tot[x][1] + fp, tot[x][2] + fn]
            d[x] = dict(tp=tp, only_second=sorted(a[x] - g[x]), only_gold=sorted(g[x] - a[x]))
        # behaviour on the reference output and on a no-op
        ref = safe_run(t["reference_code"], before)
        d["accepts_reference"] = outcome(verify(s["contract"], before, ref.df_out)[0]) == "ACCEPT" if ref.ran_ok else None
        d["rejects_noop"] = outcome(verify(s["contract"], before, before.copy())[0]) != "ACCEPT"
        # decisions on the task's runnable LLM candidates
        pairs, wrong_acc, right_rej, n_c = [], 0, 0, 0
        for c in pool:
            if c["task_id"] != t["id"] or not c["ran_ok"]:
                continue
            with open(args.pool.parent / c["output_file"], "rb") as fh:
                after = pickle.load(fh)
            sec = outcome(verify(s["contract"], before, after)[0]) == "ACCEPT"
            pairs.append((oracle.get(c["candidate_id"]) == "ACCEPT", sec))
            n_c += 1
            wrong_acc += sec and not c["is_correct"]
            right_rej += (not sec) and c["is_correct"]
        all_pairs += pairs
        d["candidates"] = dict(n=n_c, decision_agreement=round(sum(x == y for x, y in pairs) / n_c, 3) if n_c else None,
                               second_accepts_wrong=wrong_acc, second_rejects_correct=right_rej)
        per_task.append(d)
        if not d["exact"] or not d["accepts_reference"] or not d["rejects_noop"] or (n_c and d["candidates"]["decision_agreement"] < 1):
            rows_md.append(d)

    by_type = defaultdict(list)
    for d in per_task:
        by_type[d["transformation_type"]].append(d)
    scen = [dict(scenario_id=s["scenario_id"], second=s["expected_outcome"],
                 gold=scen_gold[s["scenario_id"]]["expected_outcome"]) for s in done_s]
    report = dict(
        n_tasks=len(per_task), n_scenarios=len(scen),
        contracts={x: prf(*tot[x]) for x in SECTIONS},
        exact_match=round(sum(d["exact"] for d in per_task) / len(per_task), 3) if per_task else None,
        accepts_reference=sum(bool(d["accepts_reference"]) for d in per_task),
        rejects_noop=sum(d["rejects_noop"] for d in per_task),
        candidate_decisions=dict(n=len(all_pairs),
                                 agreement=round(sum(x == y for x, y in all_pairs) / len(all_pairs), 3) if all_pairs else None,
                                 kappa=kappa(all_pairs),
                                 second_accepts_wrong=sum(d["candidates"]["second_accepts_wrong"] for d in per_task),
                                 second_rejects_correct=sum(d["candidates"]["second_rejects_correct"] for d in per_task)),
        by_type={k: dict(n=len(v), **{x: prf(sum(d[x]["tp"] for d in v), sum(len(d[x]["only_second"]) for d in v),
                                             sum(len(d[x]["only_gold"]) for d in v)) for x in SECTIONS})
                 for k, v in sorted(by_type.items())},
        scenarios=dict(agreement=round(sum(r["second"] == r["gold"] for r in scen) / len(scen), 3) if scen else None,
                       rows=scen),
        targets=dict(contract_f1=0.85, scenario_agreement=0.90),
        per_task=per_task, not_ready=problems)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agreement.json").write_text(json.dumps(report, indent=2, default=str))

    f = lambda x: "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)          # noqa: E731
    c = report["contracts"]
    md = [f"# Annotation agreement (guide §12.3)", "",
          f"Generated by `eval/annotation_agreement.py` from {len(per_task)} task sheets and {len(scen)} scenario sheets. "
          "Numbers are BEFORE any fix (report these).", "",
          "| Measure | Value | Target |", "|---|---|---|",
          f"| must_change F1 (P / R) | {f(c['must_change']['f1'])} ({f(c['must_change']['precision'])} / {f(c['must_change']['recall'])}) | ≥ 0.85 |",
          f"| must_preserve F1 (P / R) | {f(c['must_preserve']['f1'])} ({f(c['must_preserve']['precision'])} / {f(c['must_preserve']['recall'])}) | ≥ 0.85 |",
          f"| Exact match | {f(report['exact_match'])} | — |",
          f"| Second contract accepts the reference output | {report['accepts_reference']}/{len(per_task)} | all |",
          f"| Second contract rejects the no-op | {report['rejects_noop']}/{len(per_task)} | all |",
          f"| Same decision as the gold contract on LLM candidates | {f(report['candidate_decisions']['agreement'])} "
          f"(kappa {f(report['candidate_decisions']['kappa'])}, n = {report['candidate_decisions']['n']}) | — |",
          f"| Scenario outcome agreement | {f(report['scenarios']['agreement'])} | ≥ 0.90 |", "",
          "## Disagreements (resolution log)", "",
          "For each row decide (guide §12.3): spec gap → fix the guide; request ambiguity → rewrite or drop the task; "
          "mistake → fix the contract (whose?). Changing a gold contract changes the oracle results: record it here.", "",
          "| Task | Only in gold | Only in second | Accepts ref | Rejects no-op | Decision agreement | Resolution |",
          "|---|---|---|---|---|---|---|"]
    for d in rows_md:
        og = len(d["must_change"]["only_gold"]) + len(d["must_preserve"]["only_gold"])
        osd = len(d["must_change"]["only_second"]) + len(d["must_preserve"]["only_second"])
        md.append(f"| {d['task_id']} | {og} | {osd} | {d['accepts_reference']} | {d['rejects_noop']} | "
                  f"{f(d['candidates']['decision_agreement'])} | |")
    for r in scen:
        if r["second"] != r["gold"]:
            md.append(f"| {r['scenario_id']} (scenario) | {r['gold']} | {r['second']} | | | | |")
    md += ["", "Full details per task (the exact differing predicates): `agreement.json` → `per_task`."]
    (out_dir / "agreement.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[6:15]))
    print(f"-> {out_dir / 'agreement.json'}, {out_dir / 'agreement.md'}")
    return report


if __name__ == "__main__":
    main()
