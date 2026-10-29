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
B1 = ROOT / "notebooks" / "baseline1_outputs"


def test_prompt_hash_matches_kaggle_run():
    assert PROMPT_HASH == "ebbe8bceae71"


def test_frozen_schema_matches_kaggle_schema():
    if not (B1 / "schema.json").exists():
        pytest.skip("Baseline 1 outputs not present")
    assert load_prompt_schema(ROOT) == json.loads((B1 / "schema.json").read_text())["datasets"]


def test_tasks_and_messages():
    tasks = load_tasks(ROOT)
    assert len(tasks) == 60 and len({t["id"] for t in tasks}) == 60
    msgs = build_messages(tasks[0], load_prompt_schema(ROOT))
    assert msgs[0]["role"] == "system" and msgs[1]["content"].endswith("Request: " + tasks[0]["request"])
    assert "Dataset: 1000 rows, 15 columns." in msgs[1]["content"]


def test_extraction_reproduces_kaggle_code_hashes():
    if not (B1 / "candidates.jsonl").exists():
        pytest.skip("Baseline 1 outputs not present")
    rows = [json.loads(l) for l in (B1 / "candidates.jsonl").read_text().splitlines()]
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
