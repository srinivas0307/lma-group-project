"""Stage 2 metrics (README §4.4, Phase 11) from eval/runner_stage2.py runs.

    python eval/stage2_metrics.py [results/stage2] [test]

Per system (every rate is over task x seed runs; ESCALATE and REJECT count as "not correct"):
  final_accuracy      final output accepted AND equal to the reference
  no_repair_accuracy  the generator's first code passed the checker AND is correct (= the same system without repair)
  repair_gain         final_accuracy - no_repair_accuracy (improvement from repair)
  repair_success      among runs whose first code was wrong: share that end correct
  unnecessary_retries among runs whose first code was correct: share where the agent made a repair attempt
  silent_error_rate   wrong among accepted;  accept / reject / escalate rates;  parse escalations
  mean_attempts       over runs that entered repair;  tool-call validity;  unsafe attempts (target 0)
Paired McNemar on final correctness for every system pair, plus the named comparisons of the plan.
Writes <root>/stage2_metrics_<split>.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.metrics import mcnemar_exact, wilson  # noqa: E402

NAMED = [("proposed", "baseline3_repair", "whole proposed pipeline"),
         ("proposed", "no_contract_to_generator", "contract given to the generator"),
         ("proposed", "gold_contract", "cost of parser errors"),
         ("baseline3_repair", "baseline2_repair", "intent vs conventional checker, both with repair")]


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def rate(k, n):
    return round(k / n, 3) if n else None


def summarize(runs):
    n = len(runs)
    acc = [r for r in runs if r["outcome"] == "ACCEPT"]
    wrong0 = [r for r in runs if r.get("initial_correct") is False]
    right0 = [r for r in runs if r.get("initial_correct") is True]
    entered = [r for r in runs if r.get("attempts", 0) > 0]
    tools = [x for r in runs for x in r.get("tool_trace", [])]
    bad = [x for x in tools if x["error_type"] in ("invalid_args", "unknown_tool")]
    fin = sum(r["final_correct"] for r in runs)
    nor = sum(bool(r.get("no_repair_correct")) for r in runs)
    out = dict(
        n_runs=n, n_tasks=len({r["task_id"] for r in runs}),
        final_accuracy=rate(fin, n), final_accuracy_ci=wilson(fin, n),
        no_repair_accuracy=rate(nor, n), repair_gain=round((fin - nor) / n, 3) if n else None,
        initial_code_accuracy=rate(len(right0), n),
        repair_success=rate(sum(r["final_correct"] for r in wrong0), len(wrong0)),
        unnecessary_retries=rate(sum(r.get("attempts", 0) > 0 for r in right0), len(right0)),
        accept=rate(len(acc), n), reject=rate(sum(r["outcome"] == "REJECT" for r in runs), n),
        escalate=rate(sum(r["outcome"] == "ESCALATE" for r in runs), n),
        parse_escalations=sum(r.get("stage_ended") == "parse" for r in runs),
        silent_error_rate=rate(sum(not r["final_correct"] for r in acc), len(acc)),
        mean_attempts=round(sum(r["attempts"] for r in entered) / len(entered), 2) if entered else None,
        tools=dict(calls=len(tools), valid_call_rate=rate(len(tools) - len(bad), len(tools)),
                   unknown_tool_calls=sum(r.get("unknown_tool_calls", 0) for r in runs)),
        safety=dict(runs_with_unsafe_attempt=sum(r.get("unsafe_attempts", 0) > 0 for r in runs),
                    initial_code_blocked=sum(bool(r.get("initial_blocked")) for r in runs)),
        mean_tokens=round(sum(r.get("tokens", 0) for r in runs) / n) if n else None,
    )
    by = defaultdict(list)
    for r in runs:
        by[r["transformation_type"]].append(r)
    out["by_type"] = {k: dict(n=len(v), final_accuracy=rate(sum(r["final_correct"] for r in v), len(v)),
                              no_repair_accuracy=rate(sum(bool(r.get("no_repair_correct")) for r in v), len(v)))
                      for k, v in sorted(by.items())}
    return out


def pair(a, b):
    ka = {(r["task_id"], r["seed"]): r["final_correct"] for r in a}
    kb = {(r["task_id"], r["seed"]): r["final_correct"] for r in b}
    common = sorted(set(ka) & set(kb))
    x = sum(ka[k] and not kb[k] for k in common)
    y = sum(kb[k] and not ka[k] for k in common)
    return dict(n=len(common), only_first=x, only_second=y, p_value=round(mcnemar_exact(x, y), 4)) if common else None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else ROOT / "results" / "stage2"
    split = argv[1] if len(argv) > 1 else "test"
    systems, invalid = {}, []
    for p in sorted(root.glob("*/runs.jsonl")):
        info = json.loads((p.parent / "run_info.json").read_text()) if (p.parent / "run_info.json").exists() else {}
        if info.get("valid_result") is False:
            invalid.append(p.parent.name)
        systems[p.parent.name] = [r for r in load_jsonl(p) if r["split"] == split]
    names = sorted(systems)
    paired = {f"{a} vs {b}": pair(systems[a], systems[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    # Named comparisons match systems by preset (folder names may carry generator / parser tags) and only compare
    # systems that used the same code generator.
    def key(rs):
        return (rs[0].get("preset"), rs[0].get("generator_model")) if rs else (None, None)
    named = {}
    for a, b, what in NAMED:
        for sa in names:
            for sb in names:
                ka, kb = key(systems[sa]), key(systems[sb])
                if ka[0] == a and kb[0] == b and ka[1] == kb[1]:
                    named[f"{sa} vs {sb} ({what})"] = pair(systems[sa], systems[sb])
    report = dict(split=split, systems={k: summarize(v) for k, v in systems.items()}, comparisons=named,
                  paired_final_correct=paired, repair_vs_no_repair={k: pair(
                      v, [dict(r, final_correct=bool(r.get("no_repair_correct"))) for r in v]) for k, v in systems.items()},
                  not_valid_results=invalid)
    (root / f"stage2_metrics_{split}.json").write_text(json.dumps(report, indent=2))
    f = lambda x: "  -  " if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)       # noqa: E731
    print(f"{'system':<28}{'final':>7}{'no rep':>7}{'gain':>7}{'rep succ':>9}{'unnec':>7}{'silent':>7}{'esc':>6}"
          f"{'parse esc':>10}{'attempts':>9}{'tool ok':>8}{'unsafe':>7}")
    for k, m in report["systems"].items():
        print(f"{k:<28}{f(m['final_accuracy']):>7}{f(m['no_repair_accuracy']):>7}{f(m['repair_gain']):>7}"
              f"{f(m['repair_success']):>9}{f(m['unnecessary_retries']):>7}{f(m['silent_error_rate']):>7}"
              f"{f(m['escalate']):>6}{m['parse_escalations']:>10}{f(m['mean_attempts']):>9}"
              f"{f(m['tools']['valid_call_rate']):>8}{m['safety']['runs_with_unsafe_attempt']:>7}")
    for k, v in named.items():
        print(f"McNemar {k}: {v}")
    if invalid:
        print(f"NOT VALID RESULTS (pipeline checks): {invalid}")
    return report


if __name__ == "__main__":
    main()
