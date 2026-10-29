"""Prompted intent parser (Baseline 3): request + dataset schema -> intent contract JSON (vocabulary v1.0).

Prompt version p1-v1. Few-shot examples are DEV tasks only (README: no test phrasings in prompts). The same
prompt, extraction and decoding (greedy, plain generation + JSON extraction + pydantic validation; README §12)
are used for every LLM parser, so prompted and fine-tuned parsers are compared fairly.
No pandas/torch imports: this module runs inside the Kaggle GPU kernel unchanged.
"""
import hashlib
import json
import re

from igt.agents.code_generator import load_prompt_schema, load_tasks  # noqa: F401  (re-exported for scripts)

PARSER_PROMPT_VERSION = "p1-v1"
FEW_SHOT_IDS = ("adult_dedup_006", "airbnb_filt_006", "airbnb_miss_006", "adult_conv_006", "adult_col_006")
HEAD_ROWS = 3

SYSTEM_PROMPT = """You turn a data-transformation request into an intent contract: a JSON object that states what the transformation must change and what it must preserve. You never write code.

Output exactly one JSON object in a ```json block:
{"contract_version": "1.0", "must_change": [...], "must_preserve": [...]}

must_change = what the request asks for. must_preserve = everything a reasonable user expects to stay the same: rows not targeted are kept, columns not mentioned are unchanged (values and dtype), values not targeted in mentioned columns are unchanged, no rows are added.

Predicates (use only these; "target" names columns of the input data unless stated):
- {"type": "row_count", "target": null, "op": OP}
- {"type": "rows_removed_where", "target": null, "condition": COND}   every input row matching COND is removed
- {"type": "rows_retained_where", "target": null, "condition": COND}  every input row matching COND is kept
- {"type": "duplicate_count", "target": [key columns] or "*", "except": [columns], "op": "eq", "value": 0, "keep": "first"|"last"}  one row per key group survives, the first/last in the original order
- {"type": "null_count", "target": column, "op": OP}
- {"type": "column_values_unchanged", "target": column | [columns], "where": COND (optional)}  values equal on surviving rows; dtype may change
- {"type": "columns_unchanged", "target": "*", "except": [columns]}  columns still exist with the same dtype family and values
- {"type": "column_dtype", "target": column, "dtype": DTYPE, "ordered": true|false (optional), "categories": [...] (optional)}
- {"type": "column_value_relation", "target": column (may be new), "fn": FN, "args": [ARG, ...], "where": COND (optional)}  after[target] == FN(args of the input row) on surviving rows
- {"type": "columns_added", "target": [new columns]}
- {"type": "columns_removed", "target": [columns]}

OP: "unchanged" | "decreased" | "increased" (relative to the input), or "eq" | "lt" | "le" | "gt" | "ge" with "value": integer.
COND: {"column": c, "op": "=="|"!="|"<"|"<="|">"|">=", "value": scalar} | {"column": c, "op": "in"|"not_in", "value": [..]} | {"column": c, "op": "is_null"|"not_null"} | {"all": [COND, ...]} | {"any": [COND, ...]} | {"not": COND}. A missing cell makes every comparison false (also "!=" and "not_in"); use "not" when missing cells must be included.
ARG: {"col": column} or {"lit": value}.
FN: identity [col] | const [lit] | null [] | add, sub, mul, div [a, b] | round [col, lit n] | clip [col, lit lo, lit hi] (null = no bound) | lower, upper, strip [col] | map [col, lit {"from": "to", ...}] (unlisted values unchanged) | cast [col, lit DTYPE].
DTYPE: "int" | "float" | "bool" | "string" | "category" | "datetime".

Use the exact column names and values shown in the data. Every must_change predicate must be something the correct transformation actually changes."""

USER_TEMPLATE = """Dataset: {n_rows} rows, {n_cols} columns.

Columns (name: dtype, missing values):
{columns}

First {k} rows (CSV):
{head}
Request: {request}"""

PROMPT_HASH = hashlib.sha256((PARSER_PROMPT_VERSION + SYSTEM_PROMPT + USER_TEMPLATE + ",".join(FEW_SHOT_IDS))
                             .encode()).hexdigest()[:12]


def _user(task, schema):
    ds = schema[task["data_file"]]
    cols = "\n".join(f"- {c['name']}: {c['dtype']}, {c['nulls']} missing" for c in ds["columns"])
    head = "\n".join(ds["head_csv"].splitlines()[: HEAD_ROWS + 1]) + "\n"
    return USER_TEMPLATE.format(n_rows=ds["n_rows"], n_cols=len(ds["columns"]), columns=cols, k=HEAD_ROWS,
                                head=head, request=task["request"])


def _strip_ids(contract):
    return {"contract_version": contract.get("contract_version", "1.0"),
            **{d: [{k: v for k, v in p.items() if k != "id"} for p in contract.get(d) or []]
               for d in ("must_change", "must_preserve")}}


def contract_to_text(contract):
    """Compact JSON, one predicate per line (the format the examples teach)."""
    c = _strip_ids(contract)
    lines = ['{"contract_version": "1.0",']
    for d in ("must_change", "must_preserve"):
        preds = [json.dumps(p, ensure_ascii=False) for p in c[d]]
        body = ",\n  ".join(preds)
        lines.append(f' "{d}": [\n  {body}\n ]' + ("," if d == "must_change" else ""))
    lines.append("}")
    return "```json\n" + "\n".join(lines) + "\n```"


def build_messages(task, schema, examples):
    """examples: list of dicts with request, data_file and gold_contract (DEV tasks only)."""
    if task.get("split") == "test" and any(e["id"] == task["id"] for e in examples):
        raise ValueError("test task used as a few-shot example")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for e in examples:
        if e["id"] == task["id"]:
            continue
        msgs += [{"role": "user", "content": _user(e, schema)},
                 {"role": "assistant", "content": contract_to_text(e["gold_contract"])}]
    msgs.append({"role": "user", "content": _user(task, schema)})
    return msgs


def few_shot_examples(tasks_with_contracts):
    by_id = {t["id"]: t for t in tasks_with_contracts}
    ex = [by_id[i] for i in FEW_SHOT_IDS]
    assert all(e["split"] == "dev" for e in ex), "few-shot examples must be dev tasks"
    return ex


# ---------------- output extraction ----------------
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.S)


def _first_json_object(text):
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def extract_contract(text):
    """-> dict(status: ok | no_json | invalid_json | invalid_contract, contract, error)."""
    from igt.contracts import ContractError, load_contract
    blocks = _FENCE.findall(text) + [text]
    raw = None
    for b in blocks:
        raw = _first_json_object(b)
        if raw:
            break
    if not raw:
        return dict(status="no_json", contract=None, error="no JSON object in the output")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return dict(status="invalid_json", contract=None, error=str(e))
    try:
        load_contract(obj)
    except ContractError as e:
        return dict(status="invalid_contract", contract=obj, error=str(e))
    return dict(status="ok", contract=obj, error=None)
