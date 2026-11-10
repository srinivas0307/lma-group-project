"""Human trajectory labels (README Phase 10 task 4, P1): sample agent runs, label every agent step, summarise.

    .venv/bin/python eval/trajectory_labels.py sample      # -> benchmark/annotation/trajectories/*.yaml
    .venv/bin/python eval/trajectory_labels.py score       # -> benchmark/annotation/trajectories_summary.{json,md}

Sample (seed 0, fixed): 30 runs of the full agent on the test split that contain at least one agent step, stratified
by category and outcome: R repaired 8, R not repaired 6, T 6, E 5, S 5 (fewer if a stratum is smaller). 10 of the 30
(2 per stratum) are marked for a second labeller (`label_b`), for agreement. Existing sheets are never overwritten.

Each agent step (one model reply and what followed it, igt.pipeline.trace_view) gets one label:
  useful     moves toward a correct result: diagnoses the real problem, a fix that passes or removes failures, a
             justified escalation
  redundant  adds nothing: repeats a tool call or code, inspects something unrelated to the failure, no progress
  wrong      makes things worse or is incorrect: a fix that breaks something that held, invents data, ignores part
             of the request, an unjustified escalation, an unsafe attempt, an invalid reply
Labellers see the request, the gold contract, the code and what each step returned. They do not see the evaluator's
label (whether the final output was correct) until they are done; it is in the sheet's `answer_key` field at the
bottom, after the steps.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.agents.intent_parser import contract_to_text  # noqa: E402
from igt.paths import AGENT_SCENARIOS, RESULTS, TASKS  # noqa: E402
from igt.pipeline.trace_view import agent_steps  # noqa: E402

OUT = ROOT / "benchmark" / "annotation" / "trajectories"
LABELS = ("useful", "redundant", "wrong")
STRATA = [("R repaired", 8), ("R not repaired", 6), ("T", 6), ("E", 5), ("S", 5)]
HEADER = """# Trajectory labels (README Phase 10). Label EVERY step under `steps` with one of: useful | redundant | wrong
#   useful     moves toward a correct result (diagnoses the real problem; a fix that passes or removes failures;
#              a justified escalation)
#   redundant  adds nothing (repeated tool call or code; inspecting something unrelated; no progress)
#   wrong      makes things worse or is incorrect (breaks something that held; invents data; ignores part of the
#              request; unjustified escalation; unsafe attempt; invalid reply)
# Step 0 is the generator's code being checked: it is not an agent decision and is not labelled.
# Fill `label` (and `label_b` only if you are the second labeller of a run marked second_labeller: yes), set
# labeller / labeller_b and status: done. Don't read `answer_key` at the bottom until you have labelled the steps.
# Check with: .venv/bin/python eval/trajectory_labels.py score
"""


class Lit(str):
    pass


def lit(code):
    """Readable block in YAML: trailing spaces would force PyYAML to quote the whole code on one line."""
    return Lit("\n".join(l.rstrip() for l in str(code).rstrip().splitlines()) + "\n")


yaml.add_representer(Lit, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def stratum(r):
    if r["category"] == "R":
        return "R repaired" if r["outcome"] == "ACCEPT" and r["final_correct"] else "R not repaired"
    return r["category"]


def task_info(tid):
    for f in list(TASKS.glob(f"*/{tid}.yaml")) + list(AGENT_SCENARIOS.glob(f"*/{tid}.yaml")):
        t = yaml.safe_load(f.read_text())
        return " ".join(str(t["request"]).split()), t.get("gold_contract") or t.get("contract")
    return None, None


def sample(args):
    folder = Path(args.runs)
    runs = {r["scenario_id"]: r for r in load_jsonl(folder / "runs.jsonl") if r["split"] == "test"}
    traces = {t["scenario_id"]: t for t in load_jsonl(folder / "traces.jsonl")}
    by = defaultdict(list)
    for sid, r in sorted(runs.items()):
        if sid in traces and len(agent_steps(traces[sid]["trace"], traces[sid].get("initial_code"))) > 1:
            by[stratum(r)].append(sid)
    rng = random.Random(0)
    picked, double = [], set()
    for name, n in STRATA:
        got = rng.sample(by[name], min(n, len(by[name])))
        picked += got
        double |= set(got[:2])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for sid in picked:
        f = out / f"{sid}.yaml"
        if f.exists():
            continue
        r, t = runs[sid], traces[sid]
        request, contract = task_info(r["task_id"])
        steps = []
        for s in agent_steps(t["trace"], t.get("initial_code"))[1:]:
            step = dict(n=s["n"], step=s["summary"], result=s["outcome"] or "")
            if s["code"] and s["kind"] in ("fix", "tool"):
                step["code"] = lit(s["code"])
            step.update(label="", **({"label_b": ""} if sid in double else {}), note="")
            steps.append(step)
        body = dict(run_id=sid, system=folder.name, category=r["category"], stratum=stratum(r),
                    second_labeller="yes" if sid in double else "no", labeller="", labeller_b="", status="todo",
                    request=request, contract=lit(contract_to_text(contract)) if contract else None,
                    first_code=lit(t.get("initial_code") or ""),
                    first_check=agent_steps(t["trace"], t.get("initial_code"))[0]["outcome"], steps=steps,
                    run_note="", answer_key=dict(outcome=r["outcome"], final_correct=r["final_correct"],
                                                 escalate_reason=r.get("escalate_reason")))
        f.write_text(HEADER + yaml.dump(body, sort_keys=False, allow_unicode=True, width=120))
        n_new += 1
    manifest = dict(source=str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder), seed=0,
                    runs=picked, second_labeller=sorted(double),
                    strata={name: [s for s in picked if stratum(runs[s]) == name] for name, _ in STRATA})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(picked)} runs ({len(double)} for a second labeller), {n_new} new sheets -> {out}")
    return manifest


def kappa(pairs, cats=LABELS):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def score(args):
    d = Path(args.sheets)
    sheets = [yaml.safe_load(f.read_text()) for f in sorted(d.glob("*.yaml"))]
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else d.parent
    problems, done = {}, []
    for s in sheets:
        p = []
        if s.get("status") != "done":
            p.append("status is not 'done'")
        else:
            if not str(s.get("labeller") or "").strip():
                p.append("labeller is empty")
            for st in s["steps"]:
                if st.get("label") not in LABELS:
                    p.append(f"step {st['n']}: label must be one of {LABELS}")
                if s.get("second_labeller") == "yes" and st.get("label_b") not in LABELS + ("",):
                    p.append(f"step {st['n']}: label_b must be one of {LABELS} or empty")
        if p:
            problems[s["run_id"]] = p
        else:
            done.append(s)
    print(f"sheets done {len(done)}/{len(sheets)}")
    for k, p in problems.items():
        if p != ["status is not 'done'"]:
            print(f"  {k}: {'; '.join(p)}")
    if args.check or not done:
        return dict(done=len(done), problems=problems)

    steps = [(s, st) for s in done for st in s["steps"]]
    kind = lambda st: st["step"].split(" ")[0] if st["step"].startswith(("calls", "proposes", "escalates")) else "invalid"  # noqa: E731

    def dist(items):
        c = Counter(st["label"] for _, st in items)
        n = sum(c.values())
        return dict(n=n, **{l: round(c[l] / n, 3) if n else None for l in LABELS})
    pairs = [(st["label"], st["label_b"]) for s, st in steps if s.get("second_labeller") == "yes" and st.get("label_b")]
    groups = lambda f: {k: dist([x for x in steps if f(x) == k]) for k in sorted({f(x) for x in steps})}   # noqa: E731
    report = dict(
        runs=len(done), steps=len(steps), overall=dist(steps),
        by_step_kind=groups(lambda x: kind(x[1])),
        by_stratum=groups(lambda x: x[0]["stratum"]),
        by_final_outcome=groups(lambda x: f"{x[0]['answer_key']['outcome']}"
                                          f"{'' if x[0]['answer_key']['final_correct'] is None else ' correct' if x[0]['answer_key']['final_correct'] else ' wrong'}"),
        agreement=dict(n_steps=len(pairs), percent=round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None,
                       kappa=kappa(pairs)),
        not_done=problems)
    (out_dir / "trajectories_summary.json").write_text(json.dumps(report, indent=2))
    f = lambda x: "-" if x is None else f"{x:.2f}"          # noqa: E731
    md = ["# Trajectory labels: summary (README Phase 10)", "",
          f"{report['runs']} runs, {report['steps']} agent steps, labelled by people (`benchmark/annotation/trajectories/`).", "",
          "| Group | Steps | Useful | Redundant | Wrong |", "|---|---|---|---|---|",
          f"| **All** | {report['overall']['n']} | {f(report['overall']['useful'])} | {f(report['overall']['redundant'])} | {f(report['overall']['wrong'])} |"]
    for title, g in (("step", report["by_step_kind"]), ("stratum", report["by_stratum"]), ("outcome", report["by_final_outcome"])):
        md += [f"| {title}: {k} | {v['n']} | {f(v['useful'])} | {f(v['redundant'])} | {f(v['wrong'])} |" for k, v in g.items()]
    a = report["agreement"]
    md += ["", f"Agreement between two labellers on {a['n_steps']} steps: {f(a['percent'])} (Cohen's kappa {f(a['kappa'])})."]
    (out_dir / "trajectories_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[4:]))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--runs", type=Path, default=RESULTS / "stage2_agent" / "agent_full")
    s.add_argument("--out", type=Path, default=OUT)
    c = sub.add_parser("score")
    c.add_argument("--sheets", type=Path, default=OUT)
    c.add_argument("--check", action="store_true")
    c.add_argument("--out-dir", type=Path, default=None, help="where the summary goes (default: the sheets' parent)")
    args = ap.parse_args(argv)
    return sample(args) if args.cmd == "sample" else score(args)


if __name__ == "__main__":
    main()
