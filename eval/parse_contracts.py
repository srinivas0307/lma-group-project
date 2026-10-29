"""Run an LLM intent parser on every task (GPU step; run on Kaggle) -> parses.jsonl with raw outputs.

Kaggle (GPU T4, internet on; 7B needs 4-bit):
    !python eval/parse_contracts.py --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --out /kaggle/working/parse_qwen7b
CPU smoke test (no model; replies are the gold contracts, so every metric should be perfect):
    python eval/parse_contracts.py --backend gold --out /tmp/parse_gold

Decoding: greedy, plain generation (README §12 decision); extraction/validation happens here too and again
on the laptop (eval/contract_metrics.py), so the raw text is what counts. Resumable.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.generate import HFBackend, append_jsonl, load_jsonl  # noqa: E402
from igt.agents.intent_parser import (PARSER_PROMPT_VERSION, PROMPT_HASH, build_messages,  # noqa: E402
                                      contract_to_text, extract_contract, few_shot_examples, load_prompt_schema)


def load_tasks_with_contracts(root):
    out = []
    for f in sorted(Path(root).glob("eval/tasks/*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        out.append(dict(id=t["id"], split=t["split"], dataset=t["dataset"], transformation_type=t["transformation_type"],
                        request=" ".join(str(t["request"]).split()), data_file=t["data_file"],
                        gold_contract=t["gold_contract"]))
    return out


class GoldBackend:
    """CPU stand-in: replies with the gold contract (pipeline test only)."""
    def __init__(self, tasks):
        self.by_request = {t["request"]: t for t in tasks}

    def info(self):
        return {}

    def generate(self, messages_list, do_sample, n=1, seed=None):
        texts = [contract_to_text(self.by_request[m[-1]["content"].rsplit("Request: ", 1)[1]]["gold_contract"])
                 for m in messages_list]
        return texts, [0] * len(texts), [len(t) // 4 for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", choices=["hf", "gold"], default="hf")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    tasks = load_tasks_with_contracts(ROOT)
    examples = few_shot_examples(tasks)
    schema = load_prompt_schema(ROOT)
    # Few-shot (dev) tasks are parsed too, leave-one-out: build_messages drops a task from its own examples.
    todo_tasks = [t for t in tasks if t["split"] in args.splits
                  and (args.datasets is None or t["dataset"] in args.datasets)][: args.limit]
    model_id = "gold" if args.backend == "gold" else args.model
    be = GoldBackend(tasks) if args.backend == "gold" else HFBackend(
        args.model, args.max_new_tokens, 0.0, None, args.dtype, load_in_4bit=args.load_in_4bit)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "parses.jsonl"
    existing = load_jsonl(path)
    if existing and {r["model"] for r in existing} != {model_id}:
        sys.exit(f"{path} has parses from another model; use a new --out")
    done = {r["task_id"] for r in existing}
    todo = [t for t in todo_tasks if t["id"] not in done]
    t_start = time.time()
    for i in range(0, len(todo), args.batch):
        batch = todo[i:i + args.batch]
        t0 = time.time()
        texts, pts, ots = be.generate([build_messages(t, schema, examples) for t in batch], do_sample=False)
        secs = (time.time() - t0) / len(batch)
        rows = []
        for t, x, p, o in zip(batch, texts, pts, ots):
            ex = extract_contract(x)
            rows.append(dict(task_id=t["id"], split=t["split"], dataset=t["dataset"],
                             transformation_type=t["transformation_type"], model=model_id,
                             parser_prompt_version=PARSER_PROMPT_VERSION, parser_prompt_hash=PROMPT_HASH,
                             decoding="greedy", max_new_tokens=args.max_new_tokens, raw_output=x,
                             parse_status=ex["status"], parse_error=ex["error"], prompt_tokens=p, output_tokens=o,
                             truncated=o >= args.max_new_tokens, gen_time_s=round(secs, 2)))
        append_jsonl(path, rows)
        print(f"parsed {i + len(batch)}/{len(todo)} ({time.time() - t_start:.0f}s) "
              f"statuses: {[r['parse_status'] for r in rows]}", flush=True)

    allrows = load_jsonl(path)
    env = dict(model=model_id, backend=args.backend, load_in_4bit=args.load_in_4bit, dtype=args.dtype,
               parser_prompt_version=PARSER_PROMPT_VERSION, parser_prompt_hash=PROMPT_HASH,
               few_shot=list(e["id"] for e in examples), decoding="greedy", max_new_tokens=args.max_new_tokens,
               batch=args.batch, splits=args.splits, datasets=args.datasets, python=platform.python_version(),
               n_parses=len(allrows), **be.info())
    (args.out / "env.json").write_text(json.dumps(env, indent=2))
    from collections import Counter
    print("parse status:", dict(Counter(r["parse_status"] for r in allrows)), "->", path)


if __name__ == "__main__":
    main()
