"""Run the tool-using intent parser (igt.agents.tool_parser) on every task -> parses.jsonl (GPU step; Kaggle).

    python eval/parse_contracts_tools.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --out /kaggle/working/parse_tools_7b
    # test without a model: every task answered by its gold contract after one profile_column call (NOT a result)
    python eval/parse_contracts_tools.py --backend gold --out /tmp/parse_tools --limit 5

The output has the same fields as eval/parse_contracts.py (task_id, split, raw_output, parse_status, ...) plus the
tool calls, so eval/score_parsers.py scores it like any other parser run (system name in system.txt, e.g.
intent_tools_7b_t1v1). Resumable; refuses to mix models or prompt versions in one folder.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.parse_contracts import load_tasks_with_contracts  # noqa: E402
from igt.agents.intent_parser import extract_contract, few_shot_examples, load_prompt_schema  # noqa: E402
from igt.agents.tool_parser import MAX_TOOL_CALLS, PROMPT_HASH, TOOL_PROMPT_VERSION, parse_with_tools  # noqa: E402
from igt.llm import LLMClient, ScriptedBackend  # noqa: E402


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def gold_script(task):
    """Pipeline check: one tool call, then the gold contract. Its numbers are not results."""
    col = next(iter(pd.read_csv(ROOT / task["data_file"], nrows=1).columns))
    return ScriptedBackend([json.dumps({"tool": "profile_column", "args": {"column": col}}),
                            "```json\n" + json.dumps(task["gold_contract"]) + "\n```"], model="gold_script")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", choices=["hf", "gold"], default="hf")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tool-calls", type=int, default=MAX_TOOL_CALLS)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    tasks = load_tasks_with_contracts(ROOT)
    examples = few_shot_examples(tasks)
    schema = load_prompt_schema(ROOT)
    todo = [t for t in tasks if t["split"] in args.splits][: args.limit]
    data = {}
    if args.backend == "hf":
        from igt.llm.hf import HFChatBackend
        hf = HFChatBackend(args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit)
        backend_for, model_id = (lambda t: hf), args.model
    else:
        backend_for, model_id = gold_script, "gold_script"
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "parses.jsonl"
    old = load_jsonl(path)
    if old and {(r["model"], r["parser_prompt_version"]) for r in old} != {(model_id, TOOL_PROMPT_VERSION)}:
        sys.exit(f"{path} has parses from another model or prompt version; use a new --out")
    done = {r["task_id"] for r in old}
    t_start = time.time()
    for i, t in enumerate(todo, 1):
        if t["id"] in done:
            continue
        if t["data_file"] not in data:
            data[t["data_file"]] = pd.read_csv(ROOT / t["data_file"])
        client = LLMClient(backend_for(t), cache_dir=args.cache_dir)
        t0 = time.time()
        res = parse_with_tools(t, schema, examples, data[t["data_file"]], client, args.max_tool_calls, args.max_new_tokens)
        ex = extract_contract(res["raw_output"])
        row = dict(task_id=t["id"], split=t["split"], dataset=t["dataset"], transformation_type=t["transformation_type"],
                   model=model_id, parser_prompt_version=TOOL_PROMPT_VERSION, parser_prompt_hash=PROMPT_HASH,
                   shots="few", adapter=None, decoding="greedy", max_new_tokens=args.max_new_tokens,
                   raw_output=res["raw_output"], parse_status=ex["status"], parse_error=ex["error"],
                   n_tool_calls=len(res["tool_calls"]), tool_calls=res["tool_calls"], llm_calls=res["llm_calls"],
                   tokens=res["tokens"], forced_answer=res["forced_answer"], gen_time_s=round(time.time() - t0, 2))
        with open(path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(f"{i}/{len(todo)} {t['id']} {ex['status']} tools={[c['tool'] for c in res['tool_calls']]} "
              f"({time.time() - t_start:.0f}s)", flush=True)
    rows = load_jsonl(path)
    env = dict(model=model_id, backend=args.backend, load_in_4bit=args.load_in_4bit, dtype=args.dtype,
               parser_prompt_version=TOOL_PROMPT_VERSION, parser_prompt_hash=PROMPT_HASH, shots="few",
               few_shot=[e["id"] for e in examples], decoding="greedy", max_new_tokens=args.max_new_tokens,
               max_tool_calls=args.max_tool_calls, splits=args.splits, n_parses=len(rows),
               valid_result=args.backend != "gold", python=platform.python_version())
    (args.out / "env.json").write_text(json.dumps(env, indent=2))
    print(f"{len(rows)} parses -> {path}")
    return rows


if __name__ == "__main__":
    main()
