"""Extension #11 (np1), offline replay: what the agent runs would have given with the no-progress stop, for k = 1, 2, 3.
No model calls. Exact under greedy decoding: every reply before the stop point is the recorded one, so the run with the
stop rule is the recorded run cut at the stop point (same rule and same code path as
src/igt/agents/repair_agent_ext.py; a real run on Kaggle confirms it: notebooks/kaggle/agent_extensions_kaggle.ipynb).

    .venv/bin/python eval/ext_replay_no_progress.py                       # results/stage2_agent/{agent_full,agent_feedback,agent_naive}
    .venv/bin/python eval/ext_replay_no_progress.py --systems agent_full --ks 1 2 3 4

Stop rule: after an evaluated repair attempt that ends FAILED (VIOLATION, crash, block) or a first repeated code, if
the last k attempts in a row made no progress (eval/agent_metrics.no_progress), the run ends as ESCALATE.
Writes results/stage2_agent_ext/np1_replay/replay.json and replay.md (the recorded runs are only read).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.ext_common import load_jsonl  # noqa: E402
from igt.agents.repair_agent_ext import no_progress  # noqa: E402
from igt.paths import AGENT_RESULTS, RESULTS, rel  # noqa: E402

OUT = RESULTS / "stage2_agent_ext" / "np1_replay"


def stop_point(trace, k):
    """-> (index of the event where the run stops, consecutive count) or (None, 0)."""
    prev, count, first, repeats = None, 0, True, 0
    for i, e in enumerate(trace):
        ev = e["event"]
        if ev == "verified":
            res = ("pass", frozenset()) if e["status"] == "PASS" else ("fail", frozenset(e.get("failed") or []))
            failed_state = e["status"] == "VIOLATION"
        elif ev in ("execute_failed", "safety_block"):
            res, failed_state = ("exec", None), True
        elif ev == "repeated_code":
            repeats += 1
            if repeats >= 2:                       # a1-v4 escalates here anyway
                return None, 0
            res, failed_state = ("repeat", None), True
        else:
            continue
        if first:
            first, prev = False, res
            continue
        count = count + 1 if no_progress(prev, res) else 0
        if res[0] != "repeat":
            prev = res
        if failed_state and count >= k:
            return i, count
    return None, 0


def replay(runs, traces, k):
    by = {(t["scenario_id"], t.get("seed", 0)): t for t in traces}
    out = []
    for r in runs:
        t = by.get((r["scenario_id"], r["seed"]))
        rr = dict(scenario_id=r["scenario_id"], category=r["category"], outcome=r["outcome"],
                  final_correct=r["final_correct"], tokens=r["tokens"], llm_calls=r["llm_calls"], stopped=False)
        if t:
            i, n = stop_point(t["trace"], k)
            if i is not None:
                after = [e for e in t["trace"][i + 1:] if e["event"] == "llm"]
                rr.update(stopped=True, outcome="ESCALATE", final_correct=None,
                          tokens=r["tokens"] - sum(e.get("tokens", 0) for e in after),
                          llm_calls=r["llm_calls"] - len(after), recorded_outcome=r["outcome"],
                          recorded_correct=r["final_correct"])
        out.append(rr)
    return out


def summarize(rows):
    def cat(c):
        return [r for r in rows if r["category"] == c]
    R, E, S = cat("R"), cat("E"), cat("S")
    stopped = [r for r in rows if r["stopped"]]
    return dict(
        n=len(rows), outcomes=dict(Counter(r["outcome"] for r in rows)),
        stopped=len(stopped), stopped_by_category=dict(Counter(r["category"] for r in stopped)),
        stopped_recorded_outcomes=dict(Counter(r["recorded_outcome"] for r in stopped)),
        repairs_lost=sum(bool(r.get("recorded_correct")) for r in stopped),
        wrong_accepts_avoided=sum(r.get("recorded_outcome") == "ACCEPT" and r.get("recorded_correct") is False
                                  for r in stopped),
        R_repaired=sum(r["final_correct"] is True for r in R), R_n=len(R),
        E_escalated=sum(r["outcome"] == "ESCALATE" for r in E), E_n=len(E),
        S_escalated=sum(r["outcome"] == "ESCALATE" for r in S), S_n=len(S),
        escalations=sum(r["outcome"] == "ESCALATE" for r in rows),
        tokens=sum(r["tokens"] for r in rows), llm_calls=sum(r["llm_calls"] for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=AGENT_RESULTS)
    ap.add_argument("--systems", nargs="+", default=["agent_full", "agent_feedback", "agent_naive"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    report, md = {}, ["# np1 offline replay (no-progress stop), from the recorded agent test runs", "",
                      "Exact under greedy decoding: each run is the recorded run cut at the stop point. "
                      "max_attempts = 3, so k = 3 only relabels REJECT as ESCALATE and saves no calls.", "",
                      "| system | k | stopped | of which recorded REJECT / ACCEPT | repairs lost | R repaired | "
                      "E escalated | escalations (all) | LLM calls | tokens | tokens saved |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in args.systems:
        runs, traces = load_jsonl(args.root / s / "runs.jsonl"), load_jsonl(args.root / s / "traces.jsonl")
        if not runs:
            print(f"skip {s}: no runs")
            continue
        base = summarize(replay(runs, [], 1))
        report[s] = dict(recorded=base, by_k={})
        md.append(f"| {s} | recorded | - | - | - | {base['R_repaired']}/{base['R_n']} | {base['E_escalated']}/{base['E_n']} "
                  f"| {base['escalations']} | {base['llm_calls']} | {base['tokens']} | - |")
        for k in args.ks:
            rows = replay(runs, traces, k)
            m = summarize(rows)
            m["tokens_saved"] = base["tokens"] - m["tokens"]
            m["llm_calls_saved"] = base["llm_calls"] - m["llm_calls"]
            report[s]["by_k"][str(k)] = dict(summary=m, stopped_runs=[r for r in rows if r["stopped"]])
            so = m["stopped_recorded_outcomes"]
            md.append(f"| {s} | {k} | {m['stopped']} | {so.get('REJECT', 0)} / {so.get('ACCEPT', 0)} | {m['repairs_lost']} | "
                      f"{m['R_repaired']}/{m['R_n']} | {m['E_escalated']}/{m['E_n']} | {m['escalations']} | {m['llm_calls']} "
                      f"| {m['tokens']} | {m['tokens_saved']} |")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "replay.json").write_text(json.dumps(report, indent=2, default=str))
    (args.out / "replay.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten {rel(args.out / 'replay.md')}")


if __name__ == "__main__":
    main()
