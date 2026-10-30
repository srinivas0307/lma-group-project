"""Turn checked examples into chat-format SFT data for the fine-tuned parsers (docs/finetune_data_guide.md §7).

    .venv/bin/python finetune/data_gen/build_sft.py finetune/data/checked.jsonl --val-datasets penguins

Each example becomes {"messages": [system, user, assistant]} with EXACTLY the input/output format the parsers
are evaluated with (src/igt/agents/intent_parser.py): same system prompt, same dataset description
(n rows, columns with dtype and missing counts, first 3 rows), same request line, and the contract written by
`contract_to_text`. The fine-tuned parser gets no few-shot examples (it learned the format).
Validation split = whole source datasets (no dataset appears in both train and val).
Writes finetune/data/sft_train.jsonl, sft_val.jsonl and sft_stats.json.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from igt.agents.intent_parser import FT_PROMPT_VERSION, PROMPTS, _user, contract_to_text  # noqa: E402


def schema_entry(df):
    """Same procedure as the frozen evaluation schema (eval/prompts/schema_b1-v1.json)."""
    return dict(n_rows=len(df), columns=[dict(name=str(c), dtype=str(df[c].dtype), nulls=int(df[c].isna().sum()))
                                         for c in df.columns], head_csv=df.head(5).to_csv(index=False))


def to_messages(ex, schema, version=FT_PROMPT_VERSION):
    task = dict(data_file=ex["data_file"], request=" ".join(ex["request"].split()))
    return [{"role": "system", "content": PROMPTS[version]},
            {"role": "user", "content": _user(task, schema)},
            {"role": "assistant", "content": contract_to_text(ex["contract"])}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checked", type=Path)
    ap.add_argument("--val-datasets", nargs="+", required=True, help="source datasets used only for validation")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "finetune" / "data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt-version", choices=sorted(PROMPTS), default=FT_PROMPT_VERSION,
                    help="system prompt to train with; must match the version used when parsing with the adapter")
    args = ap.parse_args()
    exs = [json.loads(l) for l in args.checked.read_text().splitlines() if l.strip()]
    schema = {f: schema_entry(pd.read_csv(ROOT / f)) for f in sorted({e["data_file"] for e in exs})}
    train, val = [], []
    for e in exs:
        rec = dict(id=e["id"], source_dataset=e["source_dataset"], transformation_type=e["transformation_type"],
                   prompt_version=args.prompt_version, messages=to_messages(e, schema, args.prompt_version))
        (val if e["source_dataset"] in args.val_datasets else train).append(rec)
    random.Random(args.seed).shuffle(train)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("sft_train.jsonl", train), ("sft_val.jsonl", val)]:
        with open(args.out_dir / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = [sum(len(m["content"]) for m in r["messages"]) for r in train + val]
    stats = dict(train=len(train), val=len(val), val_datasets=args.val_datasets, prompt_version=args.prompt_version,
                 train_by_type=dict(Counter(r["transformation_type"] for r in train)),
                 train_by_dataset=dict(Counter(r["source_dataset"] for r in train)),
                 max_chars=max(chars) if chars else 0, approx_max_tokens=int(max(chars) / 3.5) if chars else 0)
    (args.out_dir / "sft_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
