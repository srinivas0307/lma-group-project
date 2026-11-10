"""Extension #7 (vllm): run the same prompts with one backend and save the replies, so transformers and vLLM can be
compared (notebooks/kaggle/vllm_check_kaggle.ipynb runs it once per backend, each in its own process).

    python eval/ext_vllm_compare.py --backend hf   --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --out hf.json
    python eval/ext_vllm_compare.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --out vllm.json
    python eval/ext_vllm_compare.py --compare hf.json vllm.json

Prompts: the Baseline 3 parser prompt (p1-v3, few-shot) for the first --n test tasks, greedy, 1024 new tokens.
The comparison reports identical replies, identical parsed contracts, and seconds per prompt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def run(args):
    from eval.parse_contracts import load_tasks_with_contracts
    from igt.agents.intent_parser import build_messages, few_shot_examples, load_prompt_schema
    from igt.llm.vllm_backend import backend_class
    tasks = load_tasks_with_contracts(ROOT)
    ex, schema = few_shot_examples(tasks), load_prompt_schema(ROOT)
    todo = [t for t in tasks if t["split"] == "test"][: args.n]
    t0 = time.time()
    be = backend_class(args.backend)(args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit, adapter=args.adapter)
    load_s = time.time() - t0
    out = []
    for t in todo:
        t1 = time.time()
        text, n_in, n_out = be.complete(build_messages(t, schema, ex, "p1-v3"), 1024, 0.0, None, 0)
        out.append(dict(task_id=t["id"], text=text, prompt_tokens=n_in, output_tokens=n_out, seconds=round(time.time() - t1, 2)))
        print(t["id"], f"{out[-1]['seconds']}s", flush=True)
    Path(args.out).write_text(json.dumps(dict(backend=args.backend, model=be.model, load_s=round(load_s, 1), replies=out), indent=2))


def compare(a_path, b_path):
    from igt.agents.intent_parser import extract_contract
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    rows = []
    for x, y in zip(a["replies"], b["replies"]):
        cx, cy = extract_contract(x["text"]), extract_contract(y["text"])
        rows.append(dict(task_id=x["task_id"], same_text=x["text"] == y["text"],
                         same_contract=cx["status"] == cy["status"] and cx["contract"] == cy["contract"],
                         s_a=x["seconds"], s_b=y["seconds"]))
    for r in rows:
        print(f"{r['task_id']:<22} same text {r['same_text']!s:<6} same contract {r['same_contract']!s:<6} "
              f"{a['backend']} {r['s_a']}s  {b['backend']} {r['s_b']}s")
    n = len(rows)
    print(f"\nidentical replies {sum(r['same_text'] for r in rows)}/{n}, identical contracts "
          f"{sum(r['same_contract'] for r in rows)}/{n}; mean s/prompt {a['backend']} "
          f"{sum(r['s_a'] for r in rows) / n:.1f}, {b['backend']} {sum(r['s_b'] for r in rows) / n:.1f}; "
          f"load {a['load_s']}s vs {b['load_s']}s")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not (args.backend and args.out):
        ap.error("--backend and --out are required (or --compare A B)")
    run(args)


if __name__ == "__main__":
    main()
