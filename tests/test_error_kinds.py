"""Draft error-kind rules (eval/draft_error_kinds.py) and the per-kind Stage 1 breakdown."""
import csv
import json
from pathlib import Path

import pandas as pd

from eval import draft_error_kinds as dk
from eval import error_kind_breakdown as ekb

ROOT = Path(__file__).resolve().parents[1]
COLS = ["a", "b", "c"]


def test_code_key_and_keep():
    assert dk.code_key("df.drop_duplicates(subset=['a', 'b'], keep='last')", COLS) == {"a", "b"}
    assert dk.code_key("df.drop_duplicates(['a'])", COLS) == {"a"}
    assert dk.code_key("df.drop_duplicates()", COLS) == set(COLS)
    assert dk.code_keep("df.drop_duplicates(subset=['a'])") == "first"
    assert dk.code_keep("df.drop_duplicates(keep=False)") == "none"


def test_classify_rules():
    before = pd.DataFrame({"a": [1, 1, 2, None], "b": [1, 2, 3, 4], "c": list("wxyz")})
    filt = dict(transformation_type="filtering", gold_contract={"must_change": [
        {"type": "rows_removed_where", "target": None, "condition": {"column": "a", "op": "<", "value": 2}}]})
    ref = before[~(before["a"] < 2)]
    assert dk.classify(filt, dict(code="df[df['a'] >= 2]"), before, ref, before[before["a"] >= 2])[0] == "wrong_null_handling"
    assert dk.classify(filt, dict(code="x"), before, ref, ref.assign(tmp=1))[0] == "helper_column_left"
    dedup = dict(transformation_type="deduplication", gold_contract={"must_change": [
        {"type": "duplicate_count", "target": ["a", "b"], "op": "eq", "value": 0, "keep": "first"}]})
    out = before.drop_duplicates(subset=["a"])
    assert dk.classify(dedup, dict(code="df.drop_duplicates(subset=['a'])"), before, before, out)[0] == "incomplete_key"
    assert dk.classify(dedup, dict(code="df.drop_duplicates(keep=False)"), before, before, out)[0] == "removes_all_copies"


def test_breakdown_matches_stage1_recall():
    rep = ekb.main([])
    m = json.loads((ROOT / "results/stage1/summary/metrics_stage1.json").read_text())["test"]
    assert rep["systems"]["Baseline 2 (conventional)"]["overall"] == m["baseline2"]["overall"]["rates"]["recall"]
    assert rep["systems"]["Oracle (gold contract)"]["overall"] == m["intent_oracle"]["overall"]["rates"]["recall"]
    with open(ROOT / "benchmark/annotation/error_kinds_pool_v4.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 272 and all(r["primary_kind"] for r in rows)
