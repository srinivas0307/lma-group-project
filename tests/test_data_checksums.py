"""The evaluation datasets are frozen inputs (data/README.md): every task, pool label and result depends on them."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMS = json.loads((ROOT / "data" / "checksums.json").read_text())["sha256"]


@pytest.mark.parametrize("name", sorted(SUMS))
def test_evaluation_data_unchanged(name):
    assert hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() == SUMS[name], \
        f"data/{name} changed: every task, pool label and result depends on it (data/README.md)"


def test_every_task_uses_a_checksummed_file():
    import yaml
    used = {yaml.safe_load(f.read_text())["data_file"] for f in (ROOT / "benchmark" / "tasks").glob("*/*.yaml")}
    assert used == {f"data/{n}" for n in SUMS}
    for f in used:
        assert len(pd.read_csv(ROOT / f)) in (891, 1000)
