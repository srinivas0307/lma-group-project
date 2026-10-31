"""Stage 2 end-to-end runs (README §4.4, Phase 11): request -> [contract] -> code -> check -> [repair] -> judged.

    # real models (Kaggle GPU); one system per call, resumable:
    python eval/runner_stage2.py --preset proposed --parses kaggle_outputs/parsing/ft_1_5b/parses.jsonl \\
        --backend hf --generator-model Qwen/Qwen2.5-3B-Instruct --agent-model Qwen/Qwen2.5-7B-Instruct --agent-4bit \\
        --cache-dir /kaggle/working/llm_cache --out results/stage2
    # pipeline check on the laptop (NOT a result: the scripted agent proposes the reference solution):
    python eval/runner_stage2.py --preset gold_contract --backend pipeline_check --limit 5 --out /tmp/stage2

Presets (one per Stage 2 system; any field can be overridden with the flags below):

    preset                     contract  generator sees contract  checker       repair
    baseline1                  none      no                       none          none
    baseline2_repair           none      no                       conventional  full agent (no contract shown)
    baseline3_repair           parsed    no                       intent        full agent
    proposed                   parsed    yes (b2-v1)              intent        full agent
    no_contract_to_generator   parsed    no                       intent        full agent
    gold_contract              oracle    yes (b2-v1)              intent        full agent

Per task and seed:
  1. Contract: gold (oracle), a parser's output (parsed: --parses parses.jsonl), or none. A required contract that is
     invalid ends the run as ESCALATE before any code is written (README §4.6, PARSE -> ESCALATED).
  2. Code generator: prompt b1-v1 (request + schema) or b2-v1 (+ contract), --generator-model.
  3. Safety check, sandbox, checker, repair: igt.agents.repair_agent.run_repair (same agent, limits and budget for
     every system; --repair-mode none = no repair). The agent model is --agent-model.
  4. Evaluator (after the run, never fed back): the initial code's output and the final accepted output are compared
     with the reference output (eval/equivalence.outputs_equal).
Writes <out>/<system>/runs.jsonl, traces.jsonl and run_info.json. Score with eval/stage2_metrics.py.
"""
import argparse
import json
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
from igt.agents.code_generator import build_generator_messages, generator_prompt_info, load_prompt_schema  # noqa: E402
from igt.agents.intent_parser import extract_contract  # noqa: E402
from igt.agents.repair_agent import AGENT_PROMPT_VERSION, AgentConfig, run_repair  # noqa: E402
from igt.baselines.conventional_checks import judge, measure  # noqa: E402
from igt.llm import LLMClient, ScriptedBackend  # noqa: E402
from igt.orchestrator import Limits  # noqa: E402
from igt.paths import RESULTS, STAGE1, TASKS  # noqa: E402
from igt.sandbox import safe_run  # noqa: E402
from igt.types import Verdict  # noqa: E402
from igt.verifier import verify  # noqa: E402

PRESETS = {
    "baseline1": dict(contract="none", generator_contract=False, verifier="none", repair_mode="none"),
    "baseline2_repair": dict(contract="none", generator_contract=False, verifier="conventional", repair_mode="full"),
    "baseline3_repair": dict(contract="parsed", generator_contract=False, verifier="intent", repair_mode="full"),
    "proposed": dict(contract="parsed", generator_contract=True, verifier="intent", repair_mode="full"),
    "no_contract_to_generator": dict(contract="parsed", generator_contract=False, verifier="intent", repair_mode="full"),
    "gold_contract": dict(contract="oracle", generator_contract=True, verifier="intent", repair_mode="full"),
}
NOOP_CODE = "```python\ndef transform(df):\n    return df.copy()\n```"


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def schema_text(entry, k=3):
    cols = "\n".join(f"- {c['name']}: {c['dtype']}, {c['nulls']} missing" for c in entry["columns"])
    head = "\n".join(entry["head_csv"].splitlines()[: k + 1])
    return f"{entry['n_rows']} rows, {len(entry['columns'])} columns.\nColumns:\n{cols}\nFirst {k} rows (CSV):\n{head}"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), required=True)
    ap.add_argument("--system", default=None, help="results folder name (default: the preset name)")
    ap.add_argument("--contract", choices=["none", "oracle", "parsed"], default=None)
    ap.add_argument("--parses", type=Path, default=None, help="parses.jsonl of the parser (contract = parsed)")
    ap.add_argument("--generator-contract", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--verifier", choices=["none", "conventional", "intent"], default=None)
    ap.add_argument("--repair-mode", choices=["none", "naive", "feedback", "full"], default=None)
    ap.add_argument("--backend", choices=["hf", "pipeline_check"], required=True)
    ap.add_argument("--generator-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--generator-4bit", action="store_true")
    ap.add_argument("--agent-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--agent-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--temperature", type=float, default=0.0, help="generator and agent")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--tasks", nargs="+", default=None, help="only these task ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-tool-calls", type=int, default=8)
    ap.add_argument("--token-budget", type=int, default=60_000)
    ap.add_argument("--generator-max-new-tokens", type=int, default=512)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=RESULTS / "stage2")
    args = ap.parse_args(argv)
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.system = args.system or args.preset
    if args.verifier == "intent" and args.contract == "none":
        ap.error("the intent checker needs a contract (--contract oracle or parsed)")
    if args.generator_contract and args.contract == "none":
        ap.error("--generator-contract needs a contract")
    if args.contract == "parsed" and not args.parses:
        ap.error("--contract parsed needs --parses <parses.jsonl>")
    return args


def config_signature(args, gen_info):
    return dict(preset=args.preset, contract=args.contract, parses=str(args.parses) if args.parses else None,
                generator_contract=args.generator_contract, verifier=args.verifier, repair_mode=args.repair_mode,
                backend=args.backend, generator_model=args.generator_model, generator_4bit=args.generator_4bit,
                agent_model=args.agent_model, agent_4bit=args.agent_4bit, temperature=args.temperature,
                generator_prompt=gen_info, agent_prompt_version=AGENT_PROMPT_VERSION,
                max_attempts=args.max_attempts, max_tool_calls=args.max_tool_calls, token_budget=args.token_budget)


def make_checker(kind, contract):
    if kind == "none":
        return lambda b, a: Verdict("PASS", [], [])
    if kind == "conventional":
        th = json.loads((STAGE1 / "baseline2" / "thresholds.json").read_text())["tuned"]["thresholds"]

        def conv(b, a):
            v = judge(measure(b, a), th)
            return Verdict(v["status"], v["failures"], v["evidence_rows"])
        return conv
    return lambda b, a, c=contract: verify(c, b, a)[0]


def first_check(trace):
    """Status of the first evaluated code (the generator's): PASS / VIOLATION / UNCERTAIN / blocked / <exec status>."""
    for e in trace:
        if e["event"] == "verified":
            return e["status"]
        if e["event"] == "safety_block":
            return "blocked"
        if e["event"] == "execute_failed":
            return e["status"]
    return None


def main(argv=None):
    args = parse_args(argv)
    tasks = []
    for f in sorted(TASKS.glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        if t["split"] in args.splits and (not args.tasks or t["id"] in args.tasks):
            tasks.append(t)
    tasks = tasks[: args.limit]
    schema = load_prompt_schema(ROOT)
    data = {f: pd.read_csv(ROOT / f) for f in {t["data_file"] for t in tasks}}
    parsed = {}
    if args.contract == "parsed":
        parsed = {r["task_id"]: extract_contract(r["raw_output"]) for r in load_jsonl(args.parses)}
    limits = Limits(max_attempts=args.max_attempts, max_tool_calls_per_attempt=args.max_tool_calls,
                    token_budget=args.token_budget)
    gen_info = generator_prompt_info(args.generator_contract)

    # ---- evaluator side: reference outputs (never passed to the runtime) ----
    ref_out = {}
    for t in tasks:
        r = safe_run(t["reference_code"], data[t["data_file"]], **limits.sandbox())
        if not r.ran_ok:
            raise SystemExit(f"reference of {t['id']} failed: {r.error}")
        ref_out[t["id"]] = r.df_out

    if args.backend == "hf":
        from igt.llm.hf import HFChatBackend
        agent_hf = HFChatBackend(args.agent_model, dtype=args.dtype, load_in_4bit=args.agent_4bit)
        same = args.generator_model == args.agent_model and args.generator_4bit == args.agent_4bit
        gen_hf = agent_hf if same else HFChatBackend(args.generator_model, dtype=args.dtype,
                                                     load_in_4bit=args.generator_4bit)
        gen_backend_for = lambda t: gen_hf                     # noqa: E731
        agent_backend_for = lambda t: agent_hf                 # noqa: E731
    else:
        def gen_backend_for(t):
            return ScriptedBackend([NOOP_CODE] * 2, model="pipeline_check_generator")

        def agent_backend_for(t):
            ref = t["reference_code"]
            if args.repair_mode in ("naive", "feedback"):
                return ScriptedBackend([f"```python\n{ref}```"] * 10, model="pipeline_check_agent")
            return ScriptedBackend([json.dumps({"action": "tool", "tool": "diff_datasets", "args": {}}),
                                    '{"action": "propose_fix"}\n```python\n' + ref + "```"] * 5,
                                   model="pipeline_check_agent")

    out_dir = args.out / args.system
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path, traces_path, info_path = out_dir / "runs.jsonl", out_dir / "traces.jsonl", out_dir / "run_info.json"
    sig = config_signature(args, gen_info)
    if info_path.exists() and load_jsonl(runs_path):
        old = json.loads(info_path.read_text()).get("config")
        if old and old != sig:
            raise SystemExit(f"{out_dir} holds runs with another configuration; use a new --system or --out\n"
                             f"old: {old}\nnew: {sig}")
    done = {(r["task_id"], r["seed"]) for r in load_jsonl(runs_path)}
    t_start = time.time()
    for seed in args.seeds:
        for i, t in enumerate(tasks, 1):
            if (t["id"], seed) in done:
                continue
            t0 = time.time()
            before = data[t["data_file"]]
            request = " ".join(str(t["request"]).split())
            rec = dict(task_id=t["id"], split=t["split"], dataset=t["dataset"], transformation_type=t["transformation_type"],
                       difficulty=t.get("difficulty"), seed=seed, system=args.system, preset=args.preset,
                       contract_source=args.contract, generator_contract=args.generator_contract, verifier=args.verifier,
                       repair_mode=args.repair_mode, generator_prompt=gen_info["version"],
                       generator_model=args.generator_model, agent_model=args.agent_model,
                       agent_prompt_version=AGENT_PROMPT_VERSION)

            # 1. contract
            contract, cstatus = None, "not_used"
            if args.contract == "oracle":
                contract, cstatus = t["gold_contract"], "ok"
            elif args.contract == "parsed":
                ex = parsed.get(t["id"])
                cstatus = "missing" if ex is None else ex["status"]
                contract = ex["contract"] if ex and ex["status"] == "ok" else None
            rec["contract_status"] = cstatus
            if args.contract != "none" and contract is None:        # README §4.6: PARSE -> ESCALATED, no code written
                rec.update(stage_ended="parse", outcome="ESCALATE", final_correct=False, initial_correct=None,
                           first_check=None, no_repair_outcome="ESCALATE", no_repair_correct=False, initial_parse_ok=None,
                           attempts=0, tool_calls=0, llm_calls=0, tokens=0, invalid_actions=0, unsafe_attempts=0,
                           unknown_tool_calls=0, initial_blocked=False, escalate_reason=f"contract {cstatus}",
                           path=["PREPARE", "PARSE", "ESCALATED"], tool_trace=[], latency_s=round(time.time() - t0, 2))
                with open(runs_path, "a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
                print(f"[seed {seed}] {i}/{len(tasks)} {t['id']} ESCALATE (contract {cstatus})", flush=True)
                continue

            # 2. code generator
            gen_client = LLMClient(gen_backend_for(t), cache_dir=args.cache_dir)
            msgs = build_generator_messages(dict(request=request, data_file=t["data_file"]), schema,
                                            contract if args.generator_contract else None)
            g = gen_client.generate(msgs, max_new_tokens=args.generator_max_new_tokens, temperature=args.temperature,
                                    seed=seed)
            ex = extract_code(g.text)
            initial = ex["code"] if ex["parse_ok"] else g.text

            # 3. check + repair (runtime: never sees the reference)
            checker = make_checker(args.verifier, contract)
            agent_client = LLMClient(agent_backend_for(t), cache_dir=args.cache_dir, token_budget=limits.token_budget)
            cfg = AgentConfig(mode=args.repair_mode, limits=limits, temperature=args.temperature)
            res = run_repair(dict(request=request), initial, before, lambda a, b=before, f=checker: f(b, a), agent_client,
                             cfg, contract_text=json.dumps(contract) if (contract and args.verifier == "intent") else None,
                             schema_text=schema_text(schema[t["data_file"]]))

            # 4. evaluator
            r0 = safe_run(initial, before, **limits.sandbox())
            initial_correct = bool(r0.ran_ok and outputs_equal(ref_out[t["id"]], r0.df_out)[0])
            final_correct = bool(res.outcome == "ACCEPT" and outputs_equal(ref_out[t["id"]], res.final_output)[0])
            fc = first_check(res.trace)
            no_repair_outcome = "ACCEPT" if fc == "PASS" else "ESCALATE" if fc == "UNCERTAIN" else "REJECT"
            rec.update(stage_ended="repair" if res.attempts else "check", outcome=res.outcome, final_correct=final_correct,
                       initial_correct=initial_correct, initial_parse_ok=ex["parse_ok"], first_check=fc,
                       no_repair_outcome=no_repair_outcome, no_repair_correct=no_repair_outcome == "ACCEPT" and initial_correct,
                       final_status=res.final_status, attempts=res.attempts, tool_calls=res.tool_calls,
                       llm_calls=res.llm_calls + gen_client.n_calls, tokens=res.tokens + gen_client.tokens_used,
                       invalid_actions=res.invalid_actions, unsafe_attempts=res.unsafe_attempts,
                       unknown_tool_calls=res.unknown_tool_calls, initial_blocked=res.initial_blocked,
                       escalate_reason=res.escalate_reason, path=["PREPARE"] + (["PARSE"] if contract else []) + res.path,
                       tool_trace=[dict(tool=e["tool"], ok=e["ok"], error_type=e["error_type"])
                                   for e in res.trace if e["event"] == "tool"],
                       latency_s=round(time.time() - t0, 2))
            with open(traces_path, "a") as fh:
                fh.write(json.dumps(dict(task_id=t["id"], seed=seed, generator_reply=g.text, initial_code=initial,
                                         trace=res.trace), default=str) + "\n")
            with open(runs_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            print(f"[seed {seed}] {i}/{len(tasks)} {t['id']} first={fc} -> {res.outcome} correct={final_correct} "
                  f"attempts={res.attempts} ({time.time() - t_start:.0f}s)", flush=True)

    info = dict(system=args.system, config=sig, valid_result=args.backend != "pipeline_check",
                splits=args.splits, seeds=args.seeds, n_runs=len(load_jsonl(runs_path)), limits=limits.to_dict(),
                python=platform.python_version())
    info_path.write_text(json.dumps(info, indent=2))
    print(f"runs in {runs_path}")
    return out_dir


if __name__ == "__main__":
    main()
