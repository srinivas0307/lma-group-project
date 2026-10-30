"""The fine-tuning example generator writes valid, self-consistent examples (fast checks, no sandbox)."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "data_gen"))


@pytest.fixture(scope="module")
def gen():
    if not (ROOT / "finetune" / "datasets" / "diamonds.csv").exists():
        pytest.skip("run finetune/data_gen/prepare_datasets.py first")
    import generate_examples
    return generate_examples


def test_every_pattern_instance_has_valid_contract_and_code(gen):
    from igt.contracts import load_contract
    random.seed(1)
    seen_types = set()
    for name in ("diamonds", "bank", "walmart"):
        ds = gen.DS(name)
        for pname, (ttype, fn) in gen.PATTERNS.items():
            for inst in fn(ds):
                c = load_contract(inst["contract"])          # raises on anything outside vocabulary v1.0
                assert c.must_change, pname
                compile(inst["code"], pname, "exec")
                assert "def transform(df)" in inst["code"]
                assert set(inst["requests"]) == {"plain", "casual", "domain", "indirect"}
                assert all(r.strip() for r in inst["requests"].values())
                seen_types.add(ttype)
    assert seen_types == {"filtering", "deduplication", "missing_values", "type_conversion", "column_transformation"}


def test_conditions_are_canonical(gen):
    c = gen.any_(gen.leaf("b", ">", 1), gen.leaf("a", "<", 2))
    assert [x["column"] for x in c["any"]] == ["a", "b"]
    assert gen.leaf("x", "is_null") == {"column": "x", "op": "is_null"}


def test_no_evaluation_dataset_is_used(gen):
    eval_files = {"adult", "airbnb", "yellow_tripdata", "titanic"}
    for name, cfg in gen.DATASETS.items():
        assert not any(e in name.lower() for e in eval_files)
    assert json.loads(json.dumps(sorted(gen.DATASETS)))      # names are plain strings
