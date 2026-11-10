"""Extension #8 (rq1) and #9+#8 (pc1rq1): re-ask a parser once when its contract is invalid (or, with --precheck, also
when a valid contract fails the data pre-check). GPU step (Kaggle: notebooks/kaggle/reask_kaggle.ipynb); only the
flagged tasks are sent to the model, every other parse is copied unchanged.

    python eval/ext_reask.py 7b_p1-v3 --backend hf --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
    python eval/ext_reask.py 7b_p1-v3 --precheck --backend hf --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
    python eval/ext_reask.py ft_1_5b --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --adapter /kaggle/input/.../adapter
    # pipeline check without a model (the second answer is the gold contract; NOT a result)
    .venv/bin/python eval/ext_reask.py 7b_p1-v3 --backend gold --out-root /tmp/reask

Input: a parse folder (name under kaggle_outputs/parsing/ or a path) with parses.jsonl and system.txt; it is only read.
Output: <out-root>/<system>_rq1/ (or _pc1rq1/) with parses.jsonl (same records plus raw_output_first, reasked,
reask_reason, reask_version, reask_tokens; raw_output = the second answer for re-asked tasks), system.txt, env.json.
With --precheck a second answer that still fails the pre-check is treated as invalid (as in pc1).
Default --out-root is results/stage1/extensions; on Kaggle use /kaggle/working/ext and copy the folder back, then
score with .venv/bin/python eval/ext_score.py results/stage1/extensions/<system>_rq1.
Each re-asked task costs one extra model call (recorded).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ext_common import EXT_STAGE1, load_jsonl, parse_folder, tasks_and_data, write_jsonl  # noqa: E402
from eval.ext_precheck import flagged_note  # noqa: E402
from igt.agents.intent_parser import extract_contract, few_shot_examples, load_prompt_schema  # noqa: E402
from igt.agents.reask import REASK_VERSION, original_messages, parser_kind, reask_turns  # noqa: E402
from igt.contracts.precheck import PRECHECK_VERSION, issues_text, precheck  # noqa: E402
from igt.llm import LLMClient, ScriptedBackend  # noqa: E402
from igt.paths import rel  # noqa: E402


def problem_of(row, task, data, use_precheck):
    """-> None (keep) or ('invalid', error) / ('precheck', issues text), plus the issues list."""
    ex = extract_contract(row["raw_output"])
    if ex["status"] != "ok":
        return ("invalid", f"{ex['status']}: {ex['error']}"), []
    if use_precheck:
        iss = precheck(ex["contract"], data[task["data_file"]])
        if iss:
            return ("precheck", issues_text(iss)), iss
    return None, []


def reask_rows(rows, tasks, data, schema, examples, backend_for, use_precheck, max_new_tokens=1024, cache_dir=None):
    out, n = [], 0
    for r in rows:
        r = dict(r, raw_output_first=r["raw_output"], reasked=False, reask_reason=None, reask_version=REASK_VERSION,
                 reask_tokens=0)
        if use_precheck:
            r["precheck_version"] = PRECHECK_VERSION
        t = tasks[r["task_id"]]
        problem, _ = problem_of(r, t, data, use_precheck)
        if problem is not None:
            n += 1
            client = LLMClient(backend_for(t), cache_dir=cache_dir)
            msgs = original_messages(r, dict(t, request=" ".join(str(t["request"]).split())), schema, examples) + \
                reask_turns(r["raw_output"], problem)       # request normalised as in eval/parse_contracts.py
            g = client.generate(msgs, max_new_tokens=max_new_tokens, temperature=0.0)
            ex2 = extract_contract(g.text)
            r.update(raw_output=g.text, reasked=True, reask_reason=problem[0], reask_problem=problem[1][:1000],
                     parse_status=ex2["status"], parse_error=ex2["error"], reask_tokens=client.tokens_used)
            if use_precheck and ex2["status"] == "ok":
                iss2 = precheck(ex2["contract"], data[t["data_file"]])
                if iss2:                                 # still doesn't fit the data: invalid, as in pc1
                    r.update(raw_output_second=g.text, raw_output=flagged_note(iss2), parse_status="precheck_flagged",
                             parse_error="; ".join(i["check"] for i in iss2), precheck_issues_second=iss2)
            print(f"re-asked {r['task_id']} ({problem[0]}) -> {r['parse_status']}", flush=True)
        out.append(r)
    return out, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="parse folder (name under kaggle_outputs/parsing/ or a path)")
    ap.add_argument("--precheck", action="store_true", help="pc1rq1: also re-ask valid contracts that fail the pre-check")
    ap.add_argument("--backend", choices=["hf", "vllm", "gold"], required=True)
    ap.add_argument("--model", default=None, help="must be the model that wrote the parses (checked)")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter of a fine-tuned parser")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=EXT_STAGE1)
    args = ap.parse_args(argv)

    d, base = parse_folder(args.folder)
    rows = load_jsonl(d / "parses.jsonl")
    tasks, data = tasks_and_data()
    from eval.parse_contracts import load_tasks_with_contracts
    examples = few_shot_examples(load_tasks_with_contracts(Path(__file__).resolve().parent.parent))
    schema = load_prompt_schema(Path(__file__).resolve().parent.parent)
    kinds = {parser_kind(r) for r in rows}
    if args.backend == "gold":
        backend_for = lambda t: ScriptedBackend(["```json\n" + json.dumps(t["gold_contract"]) + "\n```"],  # noqa: E731
                                                model="gold_reask")
        model_id = "gold_reask"
    else:
        recorded = sorted({str(r["model"]).split("+adapter:")[0] for r in rows})
        if args.model is None and len(recorded) == 1:
            args.model = recorded[0]
        if args.model is None or (recorded and args.model not in recorded):
            raise SystemExit(f"--model {args.model} is not the model that wrote these parses ({recorded})")
        if kinds & {"ft", "tools_ft"} and not args.adapter:
            raise SystemExit("these parses come from a fine-tuned parser: pass --adapter")
        from igt.llm.vllm_backend import backend_class
        be = backend_class(args.backend)(args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit, adapter=args.adapter)
        backend_for, model_id = (lambda t: be), be.model
    tag = "pc1rq1" if args.precheck else "rq1"
    system = f"{base}_{tag}"
    out = args.out_root / system
    t0 = time.time()
    new_rows, n = reask_rows(rows, tasks, data, schema, examples, backend_for, args.precheck, args.max_new_tokens,
                             args.cache_dir)
    write_jsonl(out / "parses.jsonl", new_rows)
    (out / "system.txt").write_text(system + "\n")
    env = dict(source=rel(d), base_system=base, extension=tag, reask_version=REASK_VERSION,
               precheck_version=PRECHECK_VERSION if args.precheck else None, backend=args.backend, model=model_id,
               adapter=str(args.adapter) if args.adapter else None, parser_kinds=sorted(kinds), n_tasks=len(rows),
               n_reasked=n, extra_llm_calls=n, extra_tokens=sum(r["reask_tokens"] for r in new_rows),
               valid_after=sum(extract_contract(r["raw_output"])["status"] == "ok" for r in new_rows),
               valid_before=sum(extract_contract(r["raw_output_first"])["status"] == "ok" for r in new_rows),
               valid_result=args.backend != "gold", seconds=round(time.time() - t0, 1),
               python=platform.python_version())
    (out / "env.json").write_text(json.dumps(env, indent=2))
    print(f"{system}: re-asked {n} of {len(rows)}; valid {env['valid_before']} -> {env['valid_after']}; -> {out}")
    return out


if __name__ == "__main__":
    main()
