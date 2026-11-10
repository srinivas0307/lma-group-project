"""Demo (README Phases 6, 9 and 12): the intent-gated pipeline from one command, printed as readable steps.

Three modes:

  task    CPU, no model. Request -> contract -> an LLM-written candidate from the frozen pool -> the intent verifier's
          report, next to the conventional checks (Baseline 2) on the same output. Shows a silent error being caught.
              .venv/bin/python eval/demo.py task titanic_dedup_002
              .venv/bin/python eval/demo.py task adult_filt_005 --candidate adult_filt_005__c02
              .venv/bin/python eval/demo.py task adult_filt_005 --contract parsed:kaggle_outputs/parsing/7b_p1-v3/parses.jsonl

  replay  CPU, no model. A recorded real run (agent test run or Stage 2): the first code is re-checked live, the agent's
          recorded steps are shown, then the final code is re-run and re-checked live and compared with the reference.
              .venv/bin/python eval/demo.py replay results/stage2_agent/agent_full R_test_005

  live    GPU (Kaggle). Any CSV and any request: parser -> contract -> code generator -> sandbox -> verifier -> repair
          agent. No reference exists, so there is no correctness label; the verifier's verdict is the system's answer.
              python eval/demo.py live --data data/Titanic-Dataset.csv --request "Remove passengers with no fare" \\
                  --parser-model Qwen/Qwen2.5-7B-Instruct --agent-model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
              python eval/demo.py live ... --parser-model Qwen/Qwen2.5-1.5B-Instruct --adapter <ft adapter dir>
          `--backend scripted` replaces the three models with fixed replies (for tests only).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.equivalence import outputs_equal  # noqa: E402
from igt.agents.code_extract import extract_code  # noqa: E402
from igt.agents.code_generator import build_generator_messages, load_prompt_schema  # noqa: E402
from igt.agents.intent_parser import (PARSER_PROMPT_VERSION, build_messages as parser_messages,  # noqa: E402
                                      contract_to_text, extract_contract, few_shot_examples)
from igt.agents.repair_agent import AgentConfig, run_repair  # noqa: E402
from igt.baselines.conventional_checks import judge, measure  # noqa: E402
from igt.llm import LLMClient, ScriptedBackend  # noqa: E402
from igt.paths import AGENT_SCENARIOS, POOL, STAGE1, TASKS  # noqa: E402
from igt.pipeline.trace_view import agent_steps  # noqa: E402
from igt.sandbox import safe_run  # noqa: E402
from igt.types import Verdict  # noqa: E402
from igt.verifier import verify  # noqa: E402

W = 100


def h(title):
    return f"\n{'=' * W}\n{title}\n{'-' * W}"


def indent(text, n=2):
    return "\n".join(" " * n + l for l in str(text).rstrip().splitlines())


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def load_task(tid):
    for f in list(TASKS.glob(f"*/{tid}.yaml")) + list(AGENT_SCENARIOS.glob(f"*/{tid}.yaml")):
        t = yaml.safe_load(f.read_text())
        t.setdefault("gold_contract", t.get("contract"))
        return t
    sys.exit(f"unknown task or scenario {tid}")


def conventional_verdict(before, after):
    th = json.loads((STAGE1 / "baseline2" / "thresholds.json").read_text())["tuned"]["thresholds"]
    v = judge(measure(before, after), th)
    return Verdict(v["status"], v["failures"], v["evidence_rows"])


def contract_for(task, source):
    if source == "gold":
        return task["gold_contract"], "gold contract (written by the annotator)"
    path = source.split(":", 1)[1]
    rows = {r["task_id"]: r for r in load_jsonl(path)}
    if task["id"] not in rows:
        sys.exit(f"no parse for {task['id']} in {path}")
    ex = extract_contract(rows[task["id"]]["raw_output"])
    return (ex["contract"] if ex["status"] == "ok" else None), f"parser output ({rows[task['id']].get('model')}), {ex['status']}"


# ---------------------------------------------------------------- task
def demo_task(args):
    t = load_task(args.id)
    before = pd.read_csv(ROOT / t["data_file"])
    pool = [c for c in load_jsonl(POOL) if c["task_id"] == t["id"]]
    if args.candidate == "missed":           # a wrong candidate that the conventional checks (Baseline 2) accepted
        b2 = {r["candidate_id"]: r["outcome"] for r in load_jsonl(STAGE1 / "baseline2" / "verdicts_baseline2.jsonl")}
        missed = [c for c in pool if c["ran_ok"] and not c["is_correct"] and b2.get(c["candidate_id"]) == "ACCEPT"]
        cand = missed[0] if missed else sys.exit("no wrong candidate of this task was accepted by the conventional checks")
    elif args.candidate:
        cand = next((c for c in pool if c["candidate_id"] == args.candidate), None) or sys.exit(f"{args.candidate} not in the pool")
    else:
        wrong = [c for c in pool if c["ran_ok"] and not c["is_correct"]]
        cand = (wrong or [c for c in pool if c["ran_ok"]] or sys.exit("no runnable candidate for this task"))[0]
    contract, how = contract_for(t, args.contract)
    print(h(f"Request ({t['id']}, {t['dataset']}: {len(before)} rows x {len(before.columns)} columns)"))
    print(indent(" ".join(str(t["request"]).split())))
    print(h(f"Intent contract: {how}"))
    print(indent(contract_to_text(contract) if contract else "INVALID: the system escalates (no check possible)"))
    print(h(f"LLM-written code: {cand['candidate_id']} (by {cand['generator_model']})"))
    print(indent(cand["code"]))
    if not cand["ran_ok"]:
        print(h("Result"))
        print(indent(f"the code did not run ({cand['status']}): every system rejects it"))
        return dict(candidate=cand["candidate_id"])
    with open(POOL.parent / cand["output_file"], "rb") as fh:
        after = pickle.load(fh)
    print(indent(f"-> output: {len(after)} rows x {len(after.columns)} columns (input {len(before)} x {len(before.columns)})"))
    iv = verify(contract, before, after)[0] if contract else Verdict.combine([dict(kind="uncertain", reason="invalid_contract", detail="")])
    cv = conventional_verdict(before, after)
    print(h("Intent verifier (this project)"))
    print(indent(iv.explain()))
    print(indent(f"-> {iv.outcome()}"))
    print(h("Conventional data-quality checks (Baseline 2), same output"))
    print(indent(cv.explain()))
    print(indent(f"-> {cv.outcome()}"))
    print(h("Answer key (evaluator only: output compared with the reference solution)"))
    print(indent(f"the candidate is {'CORRECT' if cand['is_correct'] else 'WRONG'}"))
    return dict(candidate=cand["candidate_id"], intent=iv.outcome(), conventional=cv.outcome(), correct=cand["is_correct"])


# ---------------------------------------------------------------- replay
def demo_replay(args):
    folder = Path(args.folder)
    key = lambda r: (r.get("scenario_id") or r.get("task_id"), r.get("seed", 0))          # noqa: E731
    runs = {key(r): r for r in load_jsonl(folder / "runs.jsonl")}
    traces = {key(r): r for r in load_jsonl(folder / "traces.jsonl")}
    run, tr = runs.get((args.id, 0)), traces.get((args.id, 0))
    if not tr:
        sys.exit(f"{args.id} not in {folder}/traces.jsonl")
    t = load_task(run["task_id"] if run else args.id)
    before = pd.read_csv(ROOT / t["data_file"])
    contract = t["gold_contract"]
    print(h(f"Recorded run {args.id} of {folder.name} ({run.get('category') or run.get('preset')}, model {run.get('model') or run.get('agent_model')})"))
    print(indent(" ".join(str(t["request"]).split())))
    if run.get("verifier", "oracle") in ("oracle", "intent") and run.get("contract_source", "oracle") in ("oracle", "gold"):
        print(h("Intent contract (gold)"))
        print(indent(contract_to_text(contract)))
    steps = agent_steps(tr["trace"], tr.get("initial_code"))
    print(h("Step 0: the first code, re-checked now"))
    print(indent(tr.get("initial_code") or ""))
    r0 = safe_run(tr.get("initial_code") or "", before)
    print(indent(verify(contract, before, r0.df_out)[0].explain() if r0.ran_ok else f"did not run: {r0.status} {r0.error}"))
    print(h("The agent's recorded steps"))
    for s in steps[1:]:
        print(indent(f"{s['n']}. {s['summary']}" + (f"  ->  {s['outcome']}" if s["outcome"] else "")))
    fixes = [s["code"] for s in steps if s["kind"] == "fix" and s["code"]]
    print(h(f"Final outcome: {run['outcome']}"))
    final = fixes[-1] if fixes else tr.get("initial_code")
    if run["outcome"] == "ACCEPT" and final:
        print(indent(final))
        r = safe_run(final, before)
        if r.ran_ok:
            print(indent(f"re-checked now: {verify(contract, before, r.df_out)[0].explain()}"))
            if t.get("reference_code"):
                ref = safe_run(t["reference_code"], before)
                ok = outputs_equal(ref.df_out, r.df_out)
                print(indent(f"answer key: {'CORRECT' if ok[0] else 'WRONG: ' + str(ok[1])}"))
    elif run.get("escalate_reason"):
        print(indent(f"reason: {run['escalate_reason']}"))
    return dict(outcome=run["outcome"], steps=len(steps))


# ---------------------------------------------------------------- live
def schema_entry(df):
    return dict(n_rows=len(df), columns=[dict(name=str(c), dtype=str(df[c].dtype), nulls=int(df[c].isna().sum()))
                                         for c in df.columns], head_csv=df.head(5).to_csv(index=False))


def demo_live(args):
    before = pd.read_csv(args.data)
    key = str(args.data)
    schema = dict(load_prompt_schema(ROOT), **{key: schema_entry(before)})
    task = dict(id="demo", split="demo", request=" ".join(args.request.split()), data_file=key)
    backends = {}

    def client(model, adapter=None, scripted=None):
        if args.backend == "scripted":
            return LLMClient(ScriptedBackend(scripted or [], model="scripted"))
        k = (model, str(adapter))
        if k not in backends:
            from igt.llm.hf import HFChatBackend
            backends[k] = HFChatBackend(model, load_in_4bit=args.load_in_4bit, adapter=adapter)
        return LLMClient(backends[k])

    print(h(f"Request ({args.data}: {len(before)} rows x {len(before.columns)} columns)"))
    print(indent(task["request"]))
    # 1. parser
    if args.adapter:
        pv = (Path(args.adapter) / "igt_prompt_version.txt").read_text().strip()
        examples = []
    else:
        from eval.parse_contracts import load_tasks_with_contracts
        pv, examples = PARSER_PROMPT_VERSION, few_shot_examples(load_tasks_with_contracts(ROOT))
    scripted_contract = ["```json\n" + Path(args.scripted_contract).read_text() + "\n```"] if args.scripted_contract else None
    p = client(args.parser_model, args.adapter, scripted_contract).generate(
        parser_messages(task, schema, examples, pv), max_new_tokens=1024)
    ex = extract_contract(p.text)
    print(h(f"Intent contract from the parser ({args.parser_model}{' + adapter' if args.adapter else ''}, prompt {pv})"))
    if ex["status"] != "ok":
        print(indent(f"{ex['status']}: {ex['error']}\n-> ESCALATE (the request goes to a human)"))
        return dict(outcome="ESCALATE", stage="parse")
    contract = ex["contract"]
    print(indent(contract_to_text(contract)))
    # 2. code generator
    scripted_code = ["```python\n" + Path(args.scripted_code).read_text() + "\n```"] if args.scripted_code else None
    g = client(args.generator_model, None, scripted_code).generate(
        build_generator_messages(task, schema, contract if args.generator_contract else None), max_new_tokens=512)
    code = extract_code(g.text)
    initial = code["code"] if code["parse_ok"] else g.text
    print(h(f"Code from the generator ({args.generator_model}, {'with' if args.generator_contract else 'without'} the contract)"))
    print(indent(initial))
    # 3. check + repair
    scripted_agent = json.loads(Path(args.scripted_agent).read_text()) if args.scripted_agent else None
    res = run_repair(dict(request=task["request"]), initial, before, lambda a: verify(contract, before, a)[0],
                     client(args.agent_model, None, scripted_agent),
                     AgentConfig(mode="full" if args.repair else "none"), contract_text=json.dumps(contract))
    print(h("Check and repair"))
    for s in agent_steps(res.trace, initial):
        print(indent(f"{s['n']}. {s['summary']}" + (f"  ->  {s['outcome']}" if s["outcome"] else "")))
    print(h(f"Final outcome: {res.outcome}"))
    if res.outcome == "ACCEPT":
        print(indent(res.final_code))
        print(indent(f"-> output {len(res.final_output)} rows x {len(res.final_output.columns)} columns"))
        if args.save_output:
            res.final_output.to_csv(args.save_output, index=False)
            print(indent(f"saved to {args.save_output}"))
    elif res.escalate_reason:
        print(indent(f"reason: {res.escalate_reason}"))
    return dict(outcome=res.outcome, attempts=res.attempts, tool_calls=res.tool_calls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("task")
    a.add_argument("id")
    a.add_argument("--candidate", default=None, help="pool candidate id, or \"missed\" = a wrong one the conventional checks accepted (default: the first runnable wrong one)")
    a.add_argument("--contract", default="gold", help="gold | parsed:<parses.jsonl>")
    b = sub.add_parser("replay")
    b.add_argument("folder")
    b.add_argument("id")
    c = sub.add_parser("live")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--request", required=True)
    c.add_argument("--backend", choices=["hf", "scripted"], default="hf")
    c.add_argument("--parser-model", default="Qwen/Qwen2.5-7B-Instruct")
    c.add_argument("--adapter", type=Path, default=None, help="fine-tuned parser adapter (zero-shot, its own prompt)")
    c.add_argument("--generator-model", default="Qwen/Qwen2.5-3B-Instruct")
    c.add_argument("--generator-contract", action=argparse.BooleanOptionalAction, default=True)
    c.add_argument("--agent-model", default="Qwen/Qwen2.5-7B-Instruct")
    c.add_argument("--repair", action=argparse.BooleanOptionalAction, default=True)
    c.add_argument("--load-in-4bit", action="store_true")
    c.add_argument("--save-output", type=Path, default=None)
    c.add_argument("--scripted-contract", type=Path, help="(scripted backend) contract JSON file")
    c.add_argument("--scripted-code", type=Path, help="(scripted backend) generator code file")
    c.add_argument("--scripted-agent", type=Path, help="(scripted backend) JSON list of agent replies")
    args = ap.parse_args(argv)
    return {"task": demo_task, "replay": demo_replay, "live": demo_live}[args.mode](args)


if __name__ == "__main__":
    main()
