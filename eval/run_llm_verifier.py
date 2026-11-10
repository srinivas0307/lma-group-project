"""Extension #3 (lv1): LLM-as-verifier baseline over the frozen pool (GPU; notebooks/kaggle/llm_verifier_kaggle.ipynb).

    python eval/run_llm_verifier.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --load-in-4bit \\
        --out /kaggle/working/ext/llm_verifier_7b_lv1 --cache-dir /kaggle/working/llm_cache
    # pipeline check without a model (accepts everything; NOT a result)
    .venv/bin/python eval/run_llm_verifier.py --backend accept_all --limit 20 --out /tmp/lv1

One model call per candidate that ran (candidates that crashed or were blocked get the standard execution-failure
VIOLATION without a call, as for every verifier). The model sees the request, the input schema and the change summary
(src/igt/baselines/llm_verifier.py); never the contract, the reference or the answer key.
Writes <out>/verdicts_<system>.jsonl (verdict format v1.0; meta: reason, malformed, tokens), system.txt and
run_info.json. Resumable. Copy the folder to results/stage1/extensions/<system>/ and score it with
    .venv/bin/python eval/ext_score.py results/stage1/extensions/llm_verifier_7b_lv1
(recall, false rejections, McNemar against Baselines 2 and 3, like every Stage 1 system).
"""
import argparse
import json
import pickle
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from igt.agents.code_generator import load_prompt_schema  # noqa: E402
from igt.baselines.llm_verifier import (LLM_VERIFIER_VERSION, PROMPT_HASH, VERIFIER_SLOT, build_messages,  # noqa: E402
                                        judge_reply)
from igt.llm import LLMClient, ScriptedBackend  # noqa: E402
from igt.paths import POOL, TASKS  # noqa: E402
from igt.types import Verdict, execution_failure, verdict_record  # noqa: E402


def schema_text(entry, k=3):
    cols = "\n".join(f"- {c['name']}: {c['dtype']}, {c['nulls']} missing" for c in entry["columns"])
    head = "\n".join(entry["head_csv"].splitlines()[: k + 1])
    return f"{entry['n_rows']} rows, {len(entry['columns'])} columns.\nColumns:\n{cols}\nFirst {k} rows (CSV):\n{head}"


class AcceptAll:
    model = "accept_all"

    def complete(self, messages, max_new_tokens, temperature, top_p, seed):
        return '{"verdict": "ACCEPT", "reason": "pipeline check"}', 0, 0


def load_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm", "accept_all"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--system", default=None, help="default llm_verifier_<size>_lv1")
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    size = args.model.split("-")[-2].lower().replace(".", "_") if args.backend != "accept_all" else "pipeline"
    system = args.system or f"llm_verifier_{size}_{LLM_VERIFIER_VERSION}"
    tasks = {}
    for f in sorted(TASKS.glob("*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        tasks[t["id"]] = t
    schema = load_prompt_schema(ROOT)
    pool = [c for c in load_jsonl(args.pool) if c["split"] in args.splits][: args.limit]
    if args.backend == "accept_all":
        backend = AcceptAll()
    else:
        from igt.llm.vllm_backend import backend_class
        backend = backend_class(args.backend)(args.model, dtype=args.dtype, load_in_4bit=args.load_in_4bit)
    client = LLMClient(backend, cache_dir=args.cache_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"verdicts_{system}.jsonl"
    info_path = args.out / "run_info.json"
    sig = dict(version=LLM_VERIFIER_VERSION, prompt_hash=PROMPT_HASH, model=backend.model, pool=str(args.pool.name))
    if info_path.exists() and load_jsonl(path) and json.loads(info_path.read_text()).get("config") != sig:
        raise SystemExit(f"{args.out} holds verdicts from another configuration; use a new --out")
    done = {r["candidate_id"] for r in load_jsonl(path)}
    data, t_start = {}, time.time()
    for i, c in enumerate(pool, 1):
        if c["candidate_id"] in done:
            continue
        meta = dict(verifier_kind="llm", version=LLM_VERIFIER_VERSION)
        if not c["ran_ok"]:
            v = Verdict.combine([execution_failure(c["status"], c.get("error"))])
        else:
            t = tasks[c["task_id"]]
            before = data.setdefault(t["data_file"], pd.read_csv(ROOT / t["data_file"]))
            with open(args.pool.parent / c["output_file"], "rb") as fh:      # frozen pool outputs (trusted, as in Stage 1)
                after = pickle.load(fh)
            n0 = client.tokens_used
            g = client.generate(build_messages(" ".join(str(t["request"]).split()), schema_text(schema[t["data_file"]]),
                                               before, after), max_new_tokens=args.max_new_tokens, temperature=0.0)
            v, j = judge_reply(g.text)
            meta.update(reason=j["reason"], malformed=j["malformed"], tokens=client.tokens_used - n0,
                        reply=g.text[:500])
        rec = verdict_record(system, VERIFIER_SLOT, c, v)
        rec["meta"] = meta
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        if i % 25 == 0 or i == len(pool):
            print(f"{i}/{len(pool)} ({time.time() - t_start:.0f}s) outcomes so far: "
                  f"{dict(Counter(r['outcome'] for r in load_jsonl(path)))}", flush=True)
    rows = load_jsonl(path)
    (args.out / "system.txt").write_text(system + "\n")
    info_path.write_text(json.dumps(dict(system=system, config=sig, backend=args.backend, splits=args.splits,
                                         n=len(rows), outcomes=dict(Counter(r["outcome"] for r in rows)),
                                         malformed=sum(bool(r["meta"].get("malformed")) for r in rows),
                                         tokens=sum(r["meta"].get("tokens", 0) for r in rows),
                                         valid_result=args.backend != "accept_all",
                                         python=platform.python_version()), indent=2))
    print(f"{len(rows)} verdicts -> {path}")
    return path


if __name__ == "__main__":
    main()
