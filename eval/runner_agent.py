"""Run one agent system over the agent scenarios (README Phase 10) and judge the results against ground truth.

    # real model (Kaggle GPU):
    python eval/runner_agent.py --system agent_full --mode full --backend hf --model Qwen/Qwen2.5-7B-Instruct \\
        --load-in-4bit --out results/stage2_agent
    # pipeline check on the laptop (NOT a result: the policy copies the reference solution):
    python eval/runner_agent.py --system smoke --mode full --backend reference_policy --limit 6 --out /tmp/agent

The runtime (repair agent) sees: request, input schema, the candidate code, the verifier's failure reports, tool
results and, for intent systems, the read-only contract. The EVALUATOR part of this script (after each run) compares
the agent's final output with the reference output; that label never goes back into the run.

Verifier (`--verifier`): oracle (gold contract), parsed:<parses.jsonl> (a parser's contract; invalid -> UNCERTAIN),
conventional (Baseline 2 tuned thresholds, results/stage1/baseline2/thresholds.json; no contract shown).

E (should escalate) and S (safety) scenarios come from benchmark/agent_scenarios/{E,S}/*.yaml. They have no pool candidate:
the initial code is written at run time by the code-generator prompt (same model, same budget), and the verifier uses
the scenario's own contract (oracle / parsed) or the conventional checks.
Writes <out>/<system>/runs.jsonl (one line per scenario x seed) and run_info.json. Resumable.
"""
import argparse
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.equivalence import outputs_equal  # noqa: E402
from igt.agents.code_extract import extract_code  # noqa: E402
from igt.agents.code_generator import build_messages as generator_messages  # noqa: E402
from igt.agents.intent_parser import extract_contract, load_prompt_schema  # noqa: E402
from igt.agents.repair_agent import AGENT_PROMPT_VERSION, AgentConfig, run_repair  # noqa: E402
from igt.baselines.conventional_checks import judge, measure  # noqa: E402
from igt.llm import LLMClient, ScriptedBackend  # noqa: E402
from igt.orchestrator import Limits  # noqa: E402
from igt.sandbox import safe_run  # noqa: E402
from igt.types import Verdict, uncertain_failure  # noqa: E402
from igt.verifier import verify  # noqa: E402
from igt.paths import POOL, RCT  # noqa: E402


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def schema_text(schema_entry, k=3):
    cols = "\n".join(f"- {c['name']}: {c['dtype']}, {c['nulls']} missing" for c in schema_entry["columns"])
    head = "\n".join(schema_entry["head_csv"].splitlines()[: k + 1])
    return f"{schema_entry['n_rows']} rows, {len(schema_entry['columns'])} columns.\nColumns:\n{cols}\nFirst {k} rows (CSV):\n{head}"


class ReferencePolicy:
    """Pipeline test only: a 'perfect' agent that inspects once, then proposes the reference solution.
    It reads ground truth, so its numbers are NOT results; run_info marks them invalid."""

    def __init__(self, refs, mode):
        self.refs, self.mode, self.model = refs, mode, "reference_policy"

    def backend_for(self, task_id):
        if task_id.startswith(("E_", "S_")):          # generated initial code, then the "right" action: escalate
            gen = "```python\ndef transform(df):\n    return df.copy()\n```"
            act = gen if self.mode in ("naive", "feedback") else json.dumps({"action": "escalate", "reason": "cannot be done"})
            return ScriptedBackend([gen] + [act] * 10, model=self.model)
        ref = self.refs[task_id]
        if self.mode in ("naive", "feedback"):
            return ScriptedBackend([f"```python\n{ref}```"] * 10, model=self.model)
        return ScriptedBackend([json.dumps({"action": "tool", "tool": "diff_datasets", "args": {}}),
                                '{"action": "propose_fix"}\n```python\n' + ref + "```"] * 5, model=self.model)


def load_es(root, dirs):
    out = []
    for d in dirs:
        for f in sorted((root / "benchmark" / "agent_scenarios" / d).glob("*.yaml")):
            sc = yaml.safe_load(f.read_text())
            sc.update(task_id=sc["id"], candidate_id=None, initial_code=None, faults=[],
                      transformation_type=sc.get("subtype", "-"))
            out.append(sc)
    return out


def build_verifier(kind, tasks, data):
    """-> function(task_id) -> (verify_fn(before, after) -> Verdict, contract_text or None)."""
    if kind == "oracle":
        return lambda tid: ((lambda b, a, c=tasks[tid]["gold_contract"]: verify(c, b, a)[0]),
                            json.dumps(tasks[tid]["gold_contract"]))
    if kind.startswith("parsed:"):
        parsed = {r["task_id"]: extract_contract(r["raw_output"]) for r in load_jsonl(kind.split(":", 1)[1])}

        def f(tid):
            if tid.startswith(("E_", "S_")):          # no parse for E/S: use the scenario contract
                return (lambda b, a, c=tasks[tid]["gold_contract"]: verify(c, b, a)[0]), json.dumps(tasks[tid]["gold_contract"])
            ex = parsed.get(tid)
            if not ex or ex["status"] != "ok":
                why = "no parse" if not ex else f"{ex['status']}: {ex['error']}"
                return (lambda b, a: Verdict.combine([uncertain_failure("invalid_contract", why)])), None
            return (lambda b, a, c=ex["contract"]: verify(c, b, a)[0]), json.dumps(ex["contract"])
        return f
    if kind == "conventional":
        th = json.loads((ROOT / "results" / "stage1" / "baseline2" / "thresholds.json").read_text())["tuned"]["thresholds"]

        def conv(b, a):
            v = judge(measure(b, a), th)
            return Verdict(v["status"], v["failures"], v["evidence_rows"])
        return lambda tid: (conv, None)
    raise SystemExit(f"unknown verifier {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, default=RCT)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--system", required=True, help="e.g. agent_full, agent_naive, agent_full_no_memory")
    ap.add_argument("--mode", choices=["none", "naive", "feedback", "full"], required=True)
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--coarse-report", action="store_true")
    ap.add_argument("--hide-contract", action="store_true")
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument("--no-faults", action="store_true", help="ignore T faults (diagnostic)")
    ap.add_argument("--verifier", default="oracle")
    ap.add_argument("--backend", choices=["hf", "reference_policy"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--categories", nargs="+", default=["R", "C", "T"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=60_000)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    scen = load_jsonl(args.scenarios) + load_es(ROOT, [c for c in ("E", "S") if c in args.categories])
    scen = [s for s in scen if s["split"] in args.splits and s["category"] in args.categories][: args.limit]
    pool = {c["candidate_id"]: c for c in load_jsonl(args.pool)}
    tasks = {}
    for f in sorted((ROOT / "benchmark" / "tasks").glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        tasks[t["id"]] = t
    for sc in scen:                                   # E/S: the scenario acts as its own task
        if sc["category"] in ("E", "S"):
            tasks[sc["id"]] = dict(id=sc["id"], request=sc["request"], data_file=sc["data_file"],
                                   gold_contract=sc["contract"], reference_code=None)
    schema = load_prompt_schema(ROOT)
    data = {f: pd.read_csv(ROOT / f) for f in {tasks[s["task_id"]]["data_file"] for s in scen}}
    get_verifier = build_verifier(args.verifier, tasks, data)
    limits = Limits(max_attempts=args.max_attempts, max_tool_calls_per_attempt=args.max_tool_calls,
                    token_budget=args.token_budget)

    # ---- evaluator side: reference outputs (never passed to the agent) ----
    refs, ref_out = {}, {}
    for tid in sorted({s["task_id"] for s in scen if s["category"] not in ("E", "S")}):
        refs[tid] = tasks[tid]["reference_code"]
        r = safe_run(refs[tid], data[tasks[tid]["data_file"]], **limits.sandbox())
        if not r.ran_ok:
            raise SystemExit(f"reference of {tid} failed: {r.error}")
        ref_out[tid] = r.df_out

    if args.backend == "hf":
        from igt.llm.hf import HFChatBackend
        hf = HFChatBackend(args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit)
        backend_for = lambda tid: hf                                  # noqa: E731
        model_name = hf.model
    else:
        rp = ReferencePolicy(refs, args.mode)
        backend_for, model_name = rp.backend_for, rp.model

    out_dir = args.out / args.system
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path, traces_path = out_dir / "runs.jsonl", out_dir / "traces.jsonl"
    old = load_jsonl(runs_path)
    stale = {r.get("agent_prompt_version", "a1-v1") for r in old} - {AGENT_PROMPT_VERSION}
    if stale:
        raise SystemExit(f"{runs_path} has runs from agent prompt {sorted(stale)}; use a new --out or --system")
    done = {(r["scenario_id"], r["seed"]) for r in old}
    t_start = time.time()
    for seed in args.seeds:
        for i, s in enumerate(scen, 1):
            if (s["id"], seed) in done:
                continue
            task = tasks[s["task_id"]]
            before = data[task["data_file"]]
            vfn, contract_text = get_verifier(s["task_id"])
            cfg = AgentConfig(mode=args.mode, memory=not args.no_memory,
                              report="coarse" if args.coarse_report else "full",
                              show_contract=not args.hide_contract, allow_escalate=not args.no_escalate,
                              limits=limits, faults=[] if args.no_faults else s["faults"],
                              temperature=args.temperature)
            client = LLMClient(backend_for(s["task_id"]), cache_dir=args.cache_dir, token_budget=limits.token_budget)
            t0 = time.time()
            initial, generated = s["initial_code"], False
            if initial is None:                           # E/S: the code generator writes the first attempt
                gtask = dict(request=" ".join(str(task["request"]).split()), data_file=task["data_file"])
                g = client.generate(generator_messages(gtask, schema), max_new_tokens=512, temperature=args.temperature,
                                    seed=seed)
                ex = extract_code(g.text)
                initial, generated = (ex["code"] if ex["parse_ok"] else g.text), True
            res = run_repair(dict(request=" ".join(str(task["request"]).split())), initial, before,
                             lambda a, b=before, f=vfn: f(b, a), client, cfg, contract_text=contract_text,
                             schema_text=schema_text(schema[task["data_file"]]))
            # ---- evaluator: judge the final output against the reference ----
            correct = None
            if res.outcome == "ACCEPT" and s["task_id"] in ref_out:
                correct = bool(outputs_equal(ref_out[s["task_id"]], res.final_output)[0])
            verified = [e for e in res.trace if e["event"] == "verified"]
            init_failed = set(verified[0]["failed"]) if verified and verified[0]["code_hash"] == res.trace[0]["code_hash"] else set()
            new_failures = sorted({x for e in verified[1:] for x in e["failed"]} - init_failed) if verified else []
            rec = dict(scenario_id=s["id"], category=s["category"], split=s["split"], task_id=s["task_id"],
                       transformation_type=s["transformation_type"], candidate_id=s["candidate_id"], seed=seed,
                       system=args.system, mode=args.mode, verifier=args.verifier, model=model_name,
                       initial_correct=bool(pool[s["candidate_id"]]["is_correct"]) if s["candidate_id"] else None,
                       initial_generated=generated, subtype=s.get("subtype"),
                       outcome=res.outcome, final_correct=correct, final_status=res.final_status,
                       attempts=res.attempts, tool_calls=res.tool_calls, llm_calls=res.llm_calls, tokens=res.tokens,
                       invalid_actions=res.invalid_actions, unsafe_attempts=res.unsafe_attempts,
                       unknown_tool_calls=res.unknown_tool_calls, escalate_reason=res.escalate_reason,
                       implicit_fixes=res.implicit_fixes, initial_blocked=res.initial_blocked,
                       agent_prompt_version=AGENT_PROMPT_VERSION,
                       faults=s["faults"], faults_fired=res.faults_fired, new_failures_after_repair=new_failures,
                       path=res.path, latency_s=round(time.time() - t0, 2),
                       tool_trace=[dict(tool=e["tool"], ok=e["ok"], error_type=e["error_type"])
                                   for e in res.trace if e["event"] == "tool"])
            with open(traces_path, "a") as fh:            # raw replies + every event, for debugging the agent
                fh.write(json.dumps(dict(scenario_id=s["id"], seed=seed, initial_code=initial, trace=res.trace),
                                    default=str) + "\n")
            with open(runs_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            print(f"[seed {seed}] {i}/{len(scen)} {s['id']} {res.outcome} correct={correct} attempts={res.attempts} "
                  f"tools={res.tool_calls} ({time.time() - t_start:.0f}s)", flush=True)

    info = dict(system=args.system, mode=args.mode, verifier=args.verifier, backend=args.backend, model=model_name,
                agent_prompt_version=AGENT_PROMPT_VERSION,
                valid_result=args.backend != "reference_policy", ablations=dict(
                    memory=not args.no_memory, report="coarse" if args.coarse_report else "full",
                    show_contract=not args.hide_contract, allow_escalate=not args.no_escalate),
                limits=limits.to_dict(), temperature=args.temperature, seeds=args.seeds, splits=args.splits,
                categories=args.categories, scenarios=str(args.scenarios), n_runs=len(load_jsonl(runs_path)),
                python=platform.python_version())
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"runs in {runs_path}")


if __name__ == "__main__":
    main()
