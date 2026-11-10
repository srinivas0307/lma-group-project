"""Print one recorded repair run as readable steps (agent test runs or Stage 2 runs). CPU only, no model.

    .venv/bin/python eval/show_trace.py results/stage2_agent/agent_full R_test_005
    .venv/bin/python eval/show_trace.py results/stage2_agent/agent_full S_011 --out case_S_011.md
    .venv/bin/python eval/show_trace.py results/stage2/proposed__gen_3b__ft_1_5b adult_filt_004

The id is a scenario id (agent runs) or a task id (Stage 2 runs).
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.paths import AGENT_SCENARIOS, TASKS  # noqa: E402
from igt.pipeline.trace_view import render_markdown  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def request_of(run):
    tid = run.get("task_id") if run else None
    for f in list(TASKS.glob(f"*/{tid}.yaml")) + list(AGENT_SCENARIOS.glob(f"*/{tid}.yaml")):
        return " ".join(str(yaml.safe_load(f.read_text())["request"]).split())
    return None


def find(folder, rid, seed=0):
    key = lambda r: (r.get("scenario_id") or r.get("task_id"), r.get("seed", 0))          # noqa: E731
    runs = {key(r): r for r in load_jsonl(Path(folder) / "runs.jsonl")}
    traces = {key(t): t for t in load_jsonl(Path(folder) / "traces.jsonl")}
    if (rid, seed) not in traces:
        sys.exit(f"{rid} (seed {seed}) not in {folder}/traces.jsonl")
    return runs.get((rid, seed)), traces[(rid, seed)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="a system folder with runs.jsonl and traces.jsonl")
    ap.add_argument("id", help="scenario id or task id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-code", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    run, trace = find(args.folder, args.id, args.seed)
    md = render_markdown(run, trace, request=request_of(run), show_code=not args.no_code)
    if args.out:
        args.out.write_text(md)
        print(f"-> {args.out}")
    else:
        print(md)
    return md


if __name__ == "__main__":
    main()
