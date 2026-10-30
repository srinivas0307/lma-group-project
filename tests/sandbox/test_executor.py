import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from igt.sandbox import run, safe_run
from igt.sandbox.executor import load_result_pickle

ROOT = Path(__file__).resolve().parents[2]
DF = pd.DataFrame({"a": [1, 2, 2, None], "b": ["x", "y", "y", "z"]}, index=[0, 1, 2, 3])
FAST = dict(timeout_s=20, cpu_s=10, mem_gb=2)


def test_runs_and_returns_dataframe_with_index():
    r = safe_run("def transform(df):\n    return df.drop_duplicates()", DF, **FAST)
    assert r.status == "ok" and r.ran_ok
    assert list(r.df_out.index) == [0, 1, 3]
    pd.testing.assert_frame_equal(r.df_out, DF.drop_duplicates())


def test_input_is_not_modified():
    before = DF.copy()
    safe_run("def transform(df):\n    df['a'] = 0\n    return df", DF, **FAST)
    pd.testing.assert_frame_equal(DF, before)


def test_custom_function_name_and_missing_function():
    assert safe_run("def clean(df):\n    return df", DF, "clean", **FAST).ran_ok
    r = safe_run("def clean(df):\n    return df", DF, **FAST)
    assert r.status == "error" and "transform" in r.error


def test_bad_return():
    r = safe_run("def transform(df):\n    return df['a']", DF, **FAST)
    assert r.status == "bad_return" and "Series" in r.error


def test_exception_is_reported():
    r = safe_run("def transform(df):\n    return df['missing_col']", DF, **FAST)
    assert r.status == "error" and r.error.startswith("KeyError")


def test_stdout_is_captured_and_truncated():
    r = safe_run("def transform(df):\n    print('hello' * 1000)\n    return df", DF, max_output_chars=50, **FAST)
    assert r.ran_ok and r.stdout.startswith("hello") and len(r.stdout) == 50


def test_blocked_code_never_runs():
    r = safe_run("import os\ndef transform(df):\n    return df", DF, **FAST)
    assert r.status == "blocked" and r.df_out is None


def test_timeout_fires():
    r = run("def transform(df):\n    while True:\n        pass", DF, timeout_s=2, cpu_s=30)
    assert r.status in {"timeout", "killed"}
    assert r.duration_s is None or r.duration_s < 10


def test_cpu_limit_fires_before_wall_timeout():
    r = run("def transform(df):\n    while True:\n        pass", DF, timeout_s=30, cpu_s=1)
    assert r.status == "killed" and ("SIGXCPU" in r.error or "SIGKILL" in r.error), r


def test_memory_limit():
    r = run("def transform(df):\n    x = np.ones((50_000, 10_000))\n    return df", DF, timeout_s=20, mem_gb=1)
    assert r.status in {"memory", "error", "killed"}


def test_rich_dtypes_round_trip():
    df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", None]), "c": pd.Categorical(["a", "b"], ordered=True),
                       "i": pd.array([1, None], dtype="Int64"), "s": pd.array(["x", None], dtype="string"),
                       "f": [1.5, np.nan], "b": [True, False]}, index=[5, 9])
    r = safe_run("def transform(df):\n    return df", df, **FAST)
    assert r.ran_ok, r.error
    pd.testing.assert_frame_equal(r.df_out, df)


def test_restricted_unpickler_rejects_code(tmp_path):
    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))
    p = tmp_path / "x.pkl"
    p.write_bytes(pickle.dumps(pd.DataFrame({"x": [Evil()]})))
    with pytest.raises(pickle.UnpicklingError):
        load_result_pickle(p)


def test_references_run_with_expected_row_counts():
    tasks = [yaml.safe_load(f.read_text()) for f in sorted((ROOT / "benchmark" / "tasks").glob("*/*.yaml"))]
    data = {}
    for t in tasks[::6]:                 # one task per (dataset, type): keeps the test fast
        df = data.setdefault(t["data_file"], pd.read_csv(ROOT / t["data_file"]))
        r = safe_run(t["reference_code"], df, **FAST)
        assert r.ran_ok, (t["id"], r.error)
        assert len(r.df_out) == t["expected_effect"]["rows_after"], t["id"]
