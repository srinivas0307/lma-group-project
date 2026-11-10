"""Training data v3-tools for the tool-using fine-tuned parsers (extension #13, prompt ft-t1; docs/extensions.md).

    .venv/bin/python finetune/data_gen/build_tools_data.py                 # -> finetune/data/v3_tools/

Built from data v2 (finetune/data/v2/checked.jsonl + its train/val split; v2 is only read):
- examples whose contract needs a value that the request doesn't state (patterns fill_median, fill_most_common)
  become multi-turn: assistant tool call profile_column(column) -> tool result computed with the real tool on the
  training dataset (igt.agents.tool_parser_ft.DataToolsFT) -> assistant final contract;
- every other example stays single-turn (system, user, contract), so the model also learns when not to call a tool.
All examples use the ft-t1 system prompt (p1-v2 + the tool instructions), zero-shot, as at parse time.
Check "the tool result matches the data": the contract's fill value must equal the median / the single most common
value in the tool result; an example that fails (or has a tie for the most common value) is dropped and reported.
--tool-repeat N repeats each tool example N times in the training file (default 1 = no upsampling).
Writes sft_train.jsonl, sft_val.jsonl (same ids and order as v2), sft_stats.json, tool_check.json.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "finetune" / "data_gen"))
from build_sft import schema_entry  # noqa: E402
from igt.agents.intent_parser import _user, contract_to_text  # noqa: E402
from igt.agents.tool_parser_ft import (FT_TOOL_PROMPT_VERSION, SYSTEM_PROMPT_FT_TOOLS, DataToolsFT,  # noqa: E402
                                       tool_call_text, tool_result_text)

V2 = ROOT / "finetune" / "data" / "v2"
OUT = ROOT / "finetune" / "data" / "v3_tools"
TOOL_PATTERNS = {"fill_median": "median", "fill_most_common": "mode"}
DATA_VERSION = "v3-tools"


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def fill_target(contract):
    p = contract["must_change"][0]
    assert p["type"] == "column_value_relation" and p["fn"] == "const", p
    return p["target"], p["args"][0]["lit"]


def same_number(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return a == b


def tool_check(kind, value, result):
    """-> None if the tool result supports the contract value, else the reason."""
    if kind == "median":
        return None if "median" in result and same_number(result["median"], value) else \
            f"median in tool result {result.get('median')!r} != contract {value!r}"
    tops = result.get("top_values") or []
    if not tops:
        return "no top values"
    best = tops[0]["count"]
    tied = [t["value"] for t in tops if t["count"] == best]
    if len(tied) > 1:
        return f"tie for the most common value: {tied[:5]}"
    return None if same_number(tied[0], value) or tied[0] == value else \
        f"most common in tool result {tied[0]!r} != contract {value!r}"


def messages_for(ex, schema, frames):
    task = dict(data_file=ex["data_file"], request=" ".join(ex["request"].split()))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_FT_TOOLS}, {"role": "user", "content": _user(task, schema)}]
    final = {"role": "assistant", "content": contract_to_text(ex["contract"])}
    if ex["pattern"] not in TOOL_PATTERNS:
        return msgs + [final], None
    col, value = fill_target(ex["contract"])
    ok, res = DataToolsFT(frames[ex["data_file"]]).call("profile_column", {"column": col})
    why = tool_check(TOOL_PATTERNS[ex["pattern"]], value, res) if ok else f"tool error {res}"
    if why:
        return None, why
    return msgs + [{"role": "assistant", "content": tool_call_text("profile_column", {"column": col})},
                   {"role": "user", "content": tool_result_text(res, 1)}, final], None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, default=V2)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--tool-repeat", type=int, default=1)
    args = ap.parse_args(argv)
    exs = {e["id"]: e for e in load_jsonl(args.v2 / "checked.jsonl")}
    split = {"train": [r["id"] for r in load_jsonl(args.v2 / "sft_train.jsonl")],
             "val": [r["id"] for r in load_jsonl(args.v2 / "sft_val.jsonl")]}
    files = sorted({exs[i]["data_file"] for ids in split.values() for i in ids})
    frames = {f: pd.read_csv(ROOT / f) for f in files}
    schema = {f: schema_entry(frames[f]) for f in files}
    out, dropped, n_tool = {"train": [], "val": []}, [], Counter()
    for name, ids in split.items():
        for i in ids:
            e = exs[i]
            msgs, why = messages_for(e, schema, frames)
            if msgs is None:
                dropped.append(dict(id=i, split=name, pattern=e["pattern"], reason=why))
                continue
            multi = len(msgs) > 3
            rec = dict(id=i, source_dataset=e["source_dataset"], transformation_type=e["transformation_type"],
                       pattern=e["pattern"], prompt_version=FT_TOOL_PROMPT_VERSION, data_version=DATA_VERSION,
                       tool_turns=int(multi), messages=msgs)
            out[name].append(rec)
            if multi:
                n_tool[name] += 1
                if name == "train":
                    out[name] += [dict(rec, id=f"{i}__rep{k}") for k in range(1, args.tool_repeat)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in out.items():
        with open(args.out_dir / f"sft_{name}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = [sum(len(m["content"]) for m in r["messages"]) for rows in out.values() for r in rows]
    stats = dict(data_version=DATA_VERSION, source="finetune/data/v2", prompt_version=FT_TOOL_PROMPT_VERSION,
                 train=len(out["train"]), val=len(out["val"]), tool_examples=dict(n_tool), tool_repeat=args.tool_repeat,
                 dropped=len(dropped), dropped_by_reason=dict(Counter(d["reason"].split(":")[0] for d in dropped)),
                 tool_patterns=sorted(TOOL_PATTERNS),
                 train_by_type=dict(Counter(r["transformation_type"] for r in out["train"])),
                 max_chars=max(chars), approx_max_tokens=int(max(chars) / 3.5))
    (args.out_dir / "sft_stats.json").write_text(json.dumps(stats, indent=2))
    (args.out_dir / "tool_check.json").write_text(json.dumps(dict(dropped=dropped), indent=2, default=str))
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
