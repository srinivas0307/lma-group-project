"""Gold contracts vs reference solutions (README Phase 2 test; guide §6.3, §6.5):
- every reference passes its own gold contract, also after reset_index() and after reversing row order;
- returning the input unchanged never passes (non-vacuity);
- every gold contract is valid under vocabulary v1.0 and only uses columns of its dataset."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
import yaml

from igt.contracts import load_contract, predicate_columns
from igt.sandbox import safe_run
from igt.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
TASKS = [yaml.safe_load(f.read_text()) for f in sorted((ROOT / "benchmark" / "tasks").glob("*/*.yaml"))]
DATA = {f: pd.read_csv(ROOT / f) for f in {t["data_file"] for t in TASKS}}


@pytest.fixture(scope="module")
def ref_outputs():
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda t: safe_run(t["reference_code"], DATA[t["data_file"]]), TASKS))
    return {t["id"]: r for t, r in zip(TASKS, res)}


@pytest.mark.parametrize("t", TASKS, ids=lambda t: t["id"])
def test_gold_contract_valid(t):
    c = load_contract(t["gold_contract"])
    cols = list(DATA[t["data_file"]].columns)
    for _, pid, p in c.predicates():
        assert all(x in cols for x in predicate_columns(p, cols)), (pid, predicate_columns(p, cols))


@pytest.mark.parametrize("t", TASKS, ids=lambda t: t["id"])
def test_reference_passes_and_noop_fails(t, ref_outputs):
    r = ref_outputs[t["id"]]
    assert r.ran_ok, r.error
    before = DATA[t["data_file"]]
    for name, out in [("reference", r.df_out), ("reset_index", r.df_out.reset_index(drop=True)),
                      ("reversed", r.df_out.iloc[::-1])]:
        v, info = verify(t["gold_contract"], before, out)
        assert v.status == "PASS", (name, info.get("alignment"), v.explain())
    v, _ = verify(t["gold_contract"], before, before.copy())
    assert v.status != "PASS", "a no-op passes the gold contract (vacuous contract)"
