"""The packaged generator prompt, code extraction and answer key must match the Baseline 1 Kaggle run exactly."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval.equivalence import outputs_equal
from igt.agents.code_extract import extract_code
from igt.agents.code_generator import PROMPT_HASH, build_messages, load_prompt_schema, load_tasks

ROOT = Path(__file__).resolve().parents[1]
B1S = [ROOT / "kaggle_outputs" / "generation" / d for d in ("3b_all_tasks", "1_5b_all_tasks")]
B1 = B1S[0]


def test_prompt_hash_matches_kaggle_run():
    assert PROMPT_HASH == "ebbe8bceae71"


def test_frozen_schema_matches_kaggle_schema():
    if not (B1 / "schema.json").exists():
        pytest.skip("Baseline 1 outputs not present")
    frozen = load_prompt_schema(ROOT)
    kaggle = json.loads((B1 / "schema.json").read_text())["datasets"]
    assert {k: frozen[k] for k in kaggle} == kaggle          # datasets added later must not change these


def test_tasks_and_messages():
    tasks = load_tasks(ROOT)
    assert len(tasks) >= 60 and len({t["id"] for t in tasks}) == len(tasks)
    schema = load_prompt_schema(ROOT)
    t0 = next(t for t in tasks if t["data_file"] in schema)
    msgs = build_messages(t0, schema)
    assert msgs[0]["role"] == "system" and msgs[1]["content"].endswith("Request: " + t0["request"])
    assert "Dataset: 1000 rows, 15 columns." in msgs[1]["content"]


def test_task_order_keeps_original_seeds():
    """New datasets must sort after adult/airbnb so the per-task sample seeds of pool v1/v2 don't change."""
    tasks = load_tasks(ROOT)
    first = [t["id"] for t in tasks[:60]]
    assert all(i.startswith(("adult_", "airbnb_")) for i in first)


def test_extraction_reproduces_kaggle_code_hashes():
    if not all((d / "candidates.jsonl").exists() for d in B1S):
        pytest.skip("Baseline 1 outputs not present")
    rows = [json.loads(l) for d in B1S for l in (d / "candidates.jsonl").read_text().splitlines()]
    for r in rows:
        ex = extract_code(r["raw_output"])
        assert ex["parse_ok"] == r["parse_ok"]
        if ex["parse_ok"]:
            assert (ex["code"], ex["code_hash"], ex["func_name"]) == (r["code"], r["code_hash"], r["func_name"])


def test_extraction_cases():
    ex = extract_code("Sure:\n```python\nimport pandas as pd\ndef transform(df):\n    return df\n\n"
                      "df = pd.read_csv('x.csv')\nprint(transform(df))\n```")
    assert ex["parse_ok"] and ex["top_level_dropped"] == 2 and "read_csv" not in ex["code"]
    assert extract_code("```python\ndef clean(df):\n    return df\n```")["renamed"]
    assert not extract_code("no code at all")["parse_ok"]
    assert extract_code("```python\ndef transform(df):\n    return df\n")["parse_ok"]     # unterminated fence


def test_outputs_equal_rules():
    ref = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert outputs_equal(ref, ref.iloc[::-1].reset_index(drop=True)[["b", "a"]])[0]       # order ignored
    assert outputs_equal(ref, ref.assign(a=[1.0000000001, 2.0]).astype({"a": "int64"}))[0]
    assert not outputs_equal(ref, ref.astype({"a": "float64"}))[0]                       # dtype family
    assert not outputs_equal(ref, ref.iloc[:1])[0]
    assert not outputs_equal(ref, ref.assign(c=1))[0]
    cat = ref.assign(b=pd.Categorical(["x", "y"], categories=["x", "y"], ordered=True))
    assert not outputs_equal(cat, ref.assign(b=pd.Categorical(["x", "y"], categories=["y", "x"], ordered=True)))[0]
    n = pd.DataFrame({"a": [np.nan, 1.0]})
    assert outputs_equal(n, pd.DataFrame({"a": [None, 1.0]}))[0]
