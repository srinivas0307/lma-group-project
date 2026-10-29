"""Generate candidate code for every task (GPU step; run on Kaggle). Same procedure as the Baseline 1 notebook.

Per task: 1 greedy generation (batched) + N sampled generations (one call with num_return_sequences=N,
seed = base seed + 1000 + task index). Writes raw replies only; labelling happens on CPU afterwards
(eval/label_candidates.py), in the pinned pandas environment.

Kaggle (GPU T4, internet on):
    !python eval/generate.py --model Qwen/Qwen2.5-1.5B-Instruct --out /kaggle/working/gen_qwen15b
Local CPU smoke test of the pipeline (no model; replies are the reference code):
    python eval/generate.py --backend reference --limit 3 --out /tmp/gen_test

Resumable: re-running with the same --out skips generations already on disk.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from igt.agents.code_generator import (PROMPT_HASH, PROMPT_VERSION, build_messages,  # noqa: E402
                                       load_prompt_schema, load_tasks, sample_seed)


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append_jsonl(p, rows):
    with open(p, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class HFBackend:
    def __init__(self, model_id, max_new_tokens, temperature, top_p, dtype):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map="auto").eval()
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p

    def info(self):
        import transformers
        t = self.torch
        return dict(torch=t.__version__, transformers=transformers.__version__,
                    gpu=t.cuda.get_device_name(0) if t.cuda.is_available() else None)

    def render(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, messages_list, do_sample, n=1, seed=None):
        t = self.torch
        with t.inference_mode():
            if seed is not None:
                t.manual_seed(seed)
                t.cuda.manual_seed_all(seed)
            enc = self.tok([self.render(m) for m in messages_list], return_tensors="pt", padding=True).to(
                self.model.device)
            kw = dict(max_new_tokens=self.max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if do_sample:
                kw.update(do_sample=True, temperature=self.temperature, top_p=self.top_p, num_return_sequences=n)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            out = self.model.generate(**enc, **kw)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(gen, skip_special_tokens=True)
            pts = enc["attention_mask"].sum(dim=1).tolist()
            pts = [p for p in pts for _ in range(n if do_sample else 1)]
            ots = [int((g != self.tok.pad_token_id).sum()) for g in gen]
        return texts, pts, ots


class ReferenceBackend:
    """CPU-only stand-in for pipeline tests: replies with the task's reference code in a fence."""
    def __init__(self, tasks):
        self.by_request = {}
        self.tasks = tasks

    def info(self):
        return {}

    def generate(self, messages_list, do_sample, n=1, seed=None):
        texts = []
        for m in messages_list:
            req = m[1]["content"].rsplit("Request: ", 1)[1]
            ref = next(t["reference_code"] for t in self.tasks if t["request"] == req)
            texts += [f"```python\n{ref}```"] * (n if do_sample else 1)
        return texts, [0] * len(texts), [len(x) // 4 for x in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--backend", choices=["hf", "reference"], default="hf")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None, help="first N tasks only (dry run)")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--greedy-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    all_tasks = load_tasks(ROOT)
    task_index = {t["id"]: i for i, t in enumerate(all_tasks)}
    tasks = [t for t in all_tasks if t["split"] in args.splits][: args.limit]
    schema = load_prompt_schema(ROOT)
    model_id = "reference" if args.backend == "reference" else args.model
    be = ReferenceBackend(all_tasks) if args.backend == "reference" else HFBackend(
        args.model, args.max_new_tokens, args.temperature, args.top_p, args.dtype)

    args.out.mkdir(parents=True, exist_ok=True)
    gen_path = args.out / "generations.jsonl"
    existing = load_jsonl(gen_path)
    if existing and {g["model"] for g in existing} != {model_id}:
        sys.exit(f"{gen_path} contains generations from another model; use a new --out")
    done = {g["gen_id"] for g in existing}

    def record(t, mode, k, seed, temp, text, pt, ot, secs):
        return dict(gen_id=f"{t['id']}__{mode}{k}", task_id=t["id"], split=t["split"], dataset=t["dataset"],
                    transformation_type=t["transformation_type"], difficulty=t["difficulty"], mode=mode,
                    sample_idx=k, seed=seed, temperature=temp, top_p=args.top_p if mode == "sample" else None,
                    model=model_id, prompt_version=PROMPT_VERSION, prompt_hash=PROMPT_HASH,
                    max_new_tokens=args.max_new_tokens, raw_output=text, prompt_tokens=pt, output_tokens=ot,
                    truncated=ot >= args.max_new_tokens, gen_time_s=round(secs, 2))

    t_start = time.time()
    todo = [t for t in tasks if f"{t['id']}__greedy0" not in done]
    for i in range(0, len(todo), args.greedy_batch):
        batch = todo[i:i + args.greedy_batch]
        t0 = time.time()
        texts, pts, ots = be.generate([build_messages(t, schema) for t in batch], do_sample=False)
        secs = (time.time() - t0) / len(batch)
        append_jsonl(gen_path, [record(t, "greedy", 0, None, 0.0, x, p, o, secs)
                                for t, x, p, o in zip(batch, texts, pts, ots)])
        print(f"greedy {i + len(batch)}/{len(todo)} ({time.time() - t_start:.0f}s)", flush=True)

    for j, t in enumerate(tasks):
        if args.n_samples == 0 or all(f"{t['id']}__sample{k}" in done for k in range(args.n_samples)):
            continue
        seed = sample_seed(args.seed, task_index[t["id"]])
        t0 = time.time()
        texts, pts, ots = be.generate([build_messages(t, schema)], do_sample=True, n=args.n_samples, seed=seed)
        secs = (time.time() - t0) / args.n_samples
        append_jsonl(gen_path, [record(t, "sample", k, seed, args.temperature, x, p, o, secs)
                                for k, (x, p, o) in enumerate(zip(texts, pts, ots))])
        print(f"sampled {j + 1}/{len(tasks)} ({time.time() - t_start:.0f}s)", flush=True)

    env = dict(model=model_id, backend=args.backend, prompt_version=PROMPT_VERSION, prompt_hash=PROMPT_HASH,
               n_samples=args.n_samples, temperature=args.temperature, top_p=args.top_p,
               max_new_tokens=args.max_new_tokens, greedy_batch=args.greedy_batch, seed=args.seed,
               splits=args.splits, limit=args.limit, dtype=args.dtype, python=platform.python_version(),
               n_generations=len(load_jsonl(gen_path)), **be.info())
    (args.out / "env.json").write_text(json.dumps(env, indent=2))
    print(f"generations on disk: {env['n_generations']} -> {gen_path}")


if __name__ == "__main__":
    main()
