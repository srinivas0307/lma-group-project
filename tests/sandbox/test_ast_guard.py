import json
from pathlib import Path

import pytest
import yaml

from igt.sandbox import check

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("code", [
    "def transform(df):\n    return df.drop_duplicates(subset=['host_id'], keep='first').copy()",
    "import numpy as np\nimport re\ndef transform(df):\n    df['x'] = np.where(df['a'] > 0, 1, 0)\n    return df",
    "from collections import Counter\ndef transform(df):\n    return df",
    "def transform(df):\n    return df[df['price'] < 500].query('price > 0')",
    "def transform(df):\n    df['c'] = pd.Categorical(df['c'], categories=['a', 'b'], ordered=True)\n    return df",
    "class Helper:\n    def __init__(self, k):\n        self.k = k\ndef transform(df):\n    return df",
])
def test_allows_normal_pandas(code):
    assert check(code).ok, check(code)


@pytest.mark.parametrize("code, reason_part", [
    ("import os", "import os"),
    ("import pandas.io.parsers", "import pandas.io"),
    ("from numpy.lib import format", "from numpy.lib import format"),
    ("from . import x", "from ."),
    ("from pandas import *", "import *"),
    ("from pandas import read_csv", "read_csv"),
    ("x = open('f')", "'open'"),
    ("x = getattr(pd, 'read_csv')", "'getattr'"),
    ("x = ().__class__", "__class__"),
    ("x = df.to_parquet('f')", "to_parquet"),
    ("x = pd.eval('1+1')", "'eval'"),
    ("x = g.gi_frame", "gi_frame"),
    ("def f():\n    global y", "global"),
    ("class A:\n    def __reduce__(self):\n        return 1", "__reduce__"),
    ("def transform(df) return df", "syntax error"),
])
def test_blocks(code, reason_part):
    r = check(code)
    assert not r.ok and reason_part in r.reason, r


def test_reports_line_number():
    r = check("def transform(df):\n    x = 1\n    import os\n    return df")
    assert not r.ok and r.lineno == 3


def test_all_pool_v1_candidates_still_pass():
    """Regression: the packaged guard must not block any candidate the Kaggle guard allowed (pool v1 had 0 blocked)."""
    pool = ROOT / "eval" / "pool_v1" / "pool_v1.jsonl"
    if not pool.exists():
        pytest.skip("pool v1 not present")
    codes = [json.loads(l)["code"] for l in pool.read_text().splitlines() if l.strip()]
    blocked = [(c, check(c).reason) for c in codes if c and not check(c).ok]
    assert not blocked, blocked[:3]


def test_all_reference_solutions_pass():
    refs = [yaml.safe_load(f.read_text())["reference_code"] for f in sorted((ROOT / "eval" / "tasks").glob("*/*.yaml"))]
    assert refs
    assert all(check(r).ok for r in refs), [check(r).reason for r in refs if not check(r).ok]
