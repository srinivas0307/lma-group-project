"""Parse every task with a tool-using FINE-TUNED parser (extension #13, ft-t1) -> parses.jsonl (GPU; Kaggle:
notebooks/kaggle/finetune_tools_kaggle.ipynb).

    python eval/parse_contracts_tools_ft.py --model Qwen/Qwen2.5-1.5B-Instruct --load-in-4bit \\
        --adapter finetune/runs/qwen25_1_5b_tools/adapter --out /kaggle/working/parse_ft_tools_1_5b
    # pipeline check without a model (gold contract after one tool call; NOT a result)
    .venv/bin/python eval/parse_contracts_tools_ft.py --backend gold --out /tmp/parse_ft_tools --limit 5

Zero-shot with the adapter's prompt version, which must be ft-t1 (finetune/train_qlora_tools.py writes it to
adapter/igt_prompt_version.txt). Same output fields as eval/parse_contracts_tools.py. After the run put the folder at
results/stage1/extensions/intent_ft_tools_<size>/ (with system.txt) and score it with eval/ext_score.py.
eval/parse_contracts_tools.py (t1-v1, few-shot 7B) is unchanged.
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
from eval.parse_contracts_tools import gold_script, load_jsonl  # noqa: E402
from igt.agents.intent_parser import extract_contract, load_prompt_schema  # noqa: E402
from igt.agents.tool_parser import MAX_TOOL_CALLS  # noqa: E402
from igt.agents.tool_parser_ft import FT_TOOL_PROMPT_VERSION, PROMPT_HASH_FT_TOOLS, parse_with_tools_ft  # noqa: E402
from igt.llm import LLMClient  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--backend", choices=["hf", "vllm", "gold"], default="hf")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--system", default=None, help="written to system.txt (default intent_ft_tools_<size>)")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tool-calls", type=int, default=MAX_TOOL_CALLS)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.backend != "gold":
        if not args.adapter:
            sys.exit("--adapter is required (a tool-using fine-tuned parser)")
        pv = (args.adapter / "igt_prompt_version.txt")
        pv = pv.read_text().strip() if pv.exists() else None
        if pv != FT_TOOL_PROMPT_VERSION:
            sys.exit(f"adapter prompt version is {pv}, expected {FT_TOOL_PROMPT_VERSION} (train with train_qlora_tools.py)")
    tasks = load_tasks_with_contracts(ROOT)
    schema = load_prompt_schema(ROOT)
    todo = [t for t in tasks if t["split"] in args.splits][: args.limit]
    size = args.model.split("-")[-2].lower().replace(".", "_")
    system = args.system or f"intent_ft_tools_{size}"
    if args.backend == "gold":
        backend_for, model_id = gold_script, "gold_script"
    else:
        from igt.llm.vllm_backend import backend_class
        be = backend_class(args.backend)(args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit,
                                         adapter=args.adapter)
        backend_for, model_id = (lambda t: be), be.model
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "parses.jsonl"
    old = load_jsonl(path)
    if old and {(r["model"], r["parser_prompt_version"]) for r in old} != {(model_id, FT_TOOL_PROMPT_VERSION)}:
        sys.exit(f"{path} has parses from another model or prompt version; use a new --out")
    done = {r["task_id"] for r in old}
    data, t_start = {}, time.time()
    for i, t in enumerate(todo, 1):
        if t["id"] in done:
            continue
        df = data.setdefault(t["data_file"], pd.read_csv(ROOT / t["data_file"]))
        client = LLMClient(backend_for(t), cache_dir=args.cache_dir)
        t0 = time.time()
        res = parse_with_tools_ft(t, schema, df, client, args.max_tool_calls, args.max_new_tokens)
        ex = extract_contract(res["raw_output"])
        row = dict(task_id=t["id"], split=t["split"], dataset=t["dataset"], transformation_type=t["transformation_type"],
                   model=model_id, parser_prompt_version=FT_TOOL_PROMPT_VERSION, parser_prompt_hash=PROMPT_HASH_FT_TOOLS,
                   shots="zero", adapter=str(args.adapter) if args.adapter else None, decoding="greedy",
                   max_new_tokens=args.max_new_tokens, raw_output=res["raw_output"], parse_status=ex["status"],
                   parse_error=ex["error"], n_tool_calls=len(res["tool_calls"]), tool_calls=res["tool_calls"],
                   llm_calls=res["llm_calls"], tokens=res["tokens"], forced_answer=res["forced_answer"],
                   gen_time_s=round(time.time() - t0, 2))
        with open(path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(f"{i}/{len(todo)} {t['id']} {ex['status']} tools={[c['tool'] for c in res['tool_calls']]} "
              f"({time.time() - t_start:.0f}s)", flush=True)
    rows = load_jsonl(path)
    (args.out / "system.txt").write_text(system + "\n")
    env = dict(model=model_id, backend=args.backend, adapter=str(args.adapter) if args.adapter else None,
               load_in_4bit=args.load_in_4bit, dtype=args.dtype, parser_prompt_version=FT_TOOL_PROMPT_VERSION,
               parser_prompt_hash=PROMPT_HASH_FT_TOOLS, shots="zero", decoding="greedy",
               max_new_tokens=args.max_new_tokens, max_tool_calls=args.max_tool_calls, splits=args.splits,
               n_parses=len(rows), valid_result=args.backend != "gold", python=platform.python_version())
    (args.out / "env.json").write_text(json.dumps(env, indent=2))
    print(f"{len(rows)} parses -> {path}")
    return rows


if __name__ == "__main__":
    main()
