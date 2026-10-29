"""Code-generator prompt (prompt version b1-v1), shared by every system and every generator model.

The generator sees only the request and the dataset schema (README G3): no contract, no row-id hints.
This module has no pandas/torch imports so it can run inside the Kaggle GPU kernel unchanged.
The dataset schema comes from the frozen file eval/prompts/schema_b1-v1.json, so prompts are
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
    return json.loads((Path(root) / "eval" / "prompts" / f"schema_{PROMPT_VERSION}.json").read_text())["datasets"]


def load_tasks(root):
    """Tasks in the canonical order (sorted file paths). The order defines the per-task seed."""
    tasks = []
    for f in sorted(Path(root).glob("eval/tasks/*/*.yaml")):
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
