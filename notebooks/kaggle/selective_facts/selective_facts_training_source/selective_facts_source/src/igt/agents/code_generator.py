"""Code-generator prompts, shared by every system and every generator model.

b1-v1: the generator sees only the request and the dataset schema (README G3): no contract, no row-id hints.
       Used for the Stage 1 pool and, in Stage 2, by Baselines 1-3 and the no-contract-to-generator ablation.
b2-v1: b1-v1 + the intent contract (Stage 2 Proposed and gold-contract systems; README §4.4).
This module has no pandas/torch imports so it can run inside the Kaggle GPU kernel unchanged.
The dataset schema comes from the frozen file benchmark/prompts/schema_b1-v1.json, so prompts are
byte-identical across runs and machines.
"""
import hashlib
import json
from pathlib import Path

import yaml

PROMPT_VERSION = "b1-v1"

SYSTEM_PROMPT = """You are an expert Python data engineer. Write pandas code that performs exactly the transformation the user asks for.

Rules:
- Define a function `transform(df)` that takes a pandas DataFrame and returns the transformed DataFrame.
- `df` is already loaded and is passed to the function. Do not read or write files and do not use the network.
- pandas (as pd) and numpy (as np) are available.
- Do not print anything.
- Reply with a single ```python code block containing the function, and nothing else."""

USER_TEMPLATE = """Dataset: {n_rows} rows, {n_cols} columns.

Columns (name: dtype, missing values):
{columns}

First 5 rows (CSV):
{head}
Request: {request}"""

PROMPT_HASH = hashlib.sha256((PROMPT_VERSION + SYSTEM_PROMPT + USER_TEMPLATE).encode()).hexdigest()[:12]
SEED_OFFSET = 1000          # sample seed = base seed + SEED_OFFSET + task index (same rule as the Kaggle run)


def load_prompt_schema(root):
    return json.loads((Path(root) / "benchmark" / "prompts" / f"schema_{PROMPT_VERSION}.json").read_text())["datasets"]


def load_tasks(root):
    """Tasks in the canonical order (sorted file paths). The order defines the per-task seed."""
    tasks = []
    for f in sorted(Path(root).glob("benchmark/tasks/*/*.yaml")):
        t = yaml.safe_load(f.read_text())
        tasks.append(dict(id=t["id"], split=t["split"], dataset=t["dataset"],
                          transformation_type=t["transformation_type"], difficulty=t.get("difficulty"),
                          request=" ".join(str(t["request"]).split()), data_file=t["data_file"],
                          reference_code=t["reference_code"]))
    return tasks


def build_messages(task, schema):
    ds = schema[task["data_file"]]
    cols = "\n".join(f"- {c['name']}: {c['dtype']}, {c['nulls']} missing" for c in ds["columns"])
    user = USER_TEMPLATE.format(n_rows=ds["n_rows"], n_cols=len(ds["columns"]), columns=cols,
                                head=ds["head_csv"], request=task["request"])
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def sample_seed(base_seed, task_index):
    return base_seed + SEED_OFFSET + task_index


# ---------------- b2-v1: contract-aware generator (Stage 2 Proposed and gold-contract systems) ----------------
# Same as b1-v1 plus (1) a short explanation of the contract in the system prompt and (2) the contract after the
# request. Nothing else differs, so "Proposed vs no-contract-to-generator ablation" measures only the effect of the
# contract. b1-v1 above is unchanged (its hash is pinned in tests). Not yet tuned on dev.
CONTRACT_PROMPT_VERSION = "b2-v1"

CONTRACT_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are also given an intent contract: the checks your output will be verified against. It has two lists:
- must_change: what the transformation must do.
- must_preserve: what must stay as it is.
The output must satisfy every predicate in both lists. Meaning of the predicates:
- row_count: number of rows compared with the input ("unchanged", "decreased", "increased") or with a number.
- rows_removed_where / rows_retained_where: every input row matching the condition is removed / kept.
- duplicate_count: duplicates on the target columns ("*" = all columns minus "except"); with "keep", exactly one row per group survives, the first or last one in the original row order.
- null_count: number of missing values in the column.
- column_values_unchanged: values stay equal on the remaining rows (the dtype may change).
- columns_unchanged: the columns keep their dtype and values ("*" = every input column minus "except").
- column_dtype: dtype family of the column (int, float, bool, string, category, datetime); for category also "ordered" and "categories".
- column_value_relation: the output column equals fn(args) computed from the input row, on rows matching "where".
- columns_added / columns_removed: these columns must appear / disappear.
Conditions are evaluated on the input data. A missing value makes every comparison false; "is_null" and "not_null" test for missing values."""

CONTRACT_USER_SUFFIX = "\n\nIntent contract (your output is checked against it):\n{contract}"

CONTRACT_PROMPT_HASH = hashlib.sha256((CONTRACT_PROMPT_VERSION + CONTRACT_SYSTEM_PROMPT + USER_TEMPLATE
                                       + CONTRACT_USER_SUFFIX).encode()).hexdigest()[:12]


def contract_for_prompt(contract):
    """Compact JSON, one predicate per line, without predicate ids (gold contracts have ids, parsed ones don't,
    so the generator sees the same format for both)."""
    lines = ['{"contract_version": "1.0",']
    for d in ("must_change", "must_preserve"):
        preds = [json.dumps({k: v for k, v in p.items() if k != "id"}, ensure_ascii=False) for p in contract.get(d) or []]
        lines.append(f' "{d}": [\n  ' + ",\n  ".join(preds) + "\n ]" + ("," if d == "must_change" else ""))
    lines.append("}")
    return "\n".join(lines)


def build_generator_messages(task, schema, contract=None):
    """contract=None -> exactly the b1-v1 messages. Otherwise the b2-v1 messages with the contract appended."""
    msgs = build_messages(task, schema)
    if contract is None:
        return msgs
    return [{"role": "system", "content": CONTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": msgs[1]["content"] + CONTRACT_USER_SUFFIX.format(contract=contract_for_prompt(contract))}]


def generator_prompt_info(with_contract):
    return dict(version=CONTRACT_PROMPT_VERSION, hash=CONTRACT_PROMPT_HASH) if with_contract else \
        dict(version=PROMPT_VERSION, hash=PROMPT_HASH)
