"""Regression tests for the bugs found in the independent code review."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import harness
import pandas as pd, numpy as np
ns = harness.load()
P = []
def check(c, m):
    print(("ok  " if c else "FAIL"), m)
    if not c: P.append(m)

df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "s": ["x", "y", "z"], "d": ["2020-01-01", "2020-01-02", "2020-01-03"]})
C = lambda mc, mp=None: ns["normalize_contract"]({"must_change": mc, "must_preserve": mp or []})

# 1 duplicate output column -> VIOLATION, no crash
c, _ = C([{"type": "row_count", "target": None, "op": "unchanged"}])
out = pd.concat([df, df[["a"]]], axis=1)
check(ns["verify"](df, out, c)["verdict"] == "VIOLATION", "duplicate column -> VIOLATION")

# 2 round on a text column -> hard parser feedback (UNCERTAIN), no exception
c, e = C([{"type": "column_value_relation", "target": "s", "fn": "round", "args": [{"col": "s"}, {"lit": 1}]}])
hard, _ = ns["contract_issues"](c, df)
check(any("cannot be evaluated" in h for h in hard), "round on text -> feedback, not crash")
c, e = C([{"type": "column_value_relation", "target": "a", "fn": "clip", "args": [{"col": "a"}, {"lit": "x"}, {"lit": None}]}])
check(any("clip bounds" in x for x in e), "clip with text bound rejected by schema")

# 3/4 contract scoring
cols = list(df.columns)
g, _ = C([{"type": "column_value_relation", "target": "a", "fn": "sub", "args": [{"col": "a"}, {"lit": 1}]}])
p, _ = C([{"type": "column_value_relation", "target": "a", "fn": "sub", "args": [{"lit": 1}, {"col": "a"}]}])
check(not ns["contract_scores"](p, g, cols)["exact_match"], "sub arg order matters in scoring")
g, _ = C([{"type": "column_dtype", "target": "s", "dtype": "category", "ordered": True, "categories": ["x", "y", "z"]}])
p, _ = C([{"type": "column_dtype", "target": "s", "dtype": "category", "ordered": True, "categories": ["z", "y", "x"]}])
check(not ns["contract_scores"](p, g, cols)["exact_match"], "ordered categories order matters")
g, _ = C([{"type": "column_dtype", "target": "s", "dtype": "category", "categories": ["x", "y", "z"]}])
p, _ = C([{"type": "column_dtype", "target": "s", "dtype": "category", "ordered": False, "categories": ["z", "y", "x"]}])
check(ns["contract_scores"](p, g, cols)["exact_match"], "unordered categories as a set + ordered default")
g, _ = C([{"type": "column_value_relation", "target": "a", "fn": "null", "args": []}])
p, _ = C([{"type": "column_value_relation", "target": "a", "fn": "null"}])
check(ns["contract_scores"](p, g, cols)["exact_match"], "missing args == default []")
g, _ = C([{"type": "rows_removed_where", "target": None, "condition": {"column": "s", "op": "in", "value": ["x", "y"]}}])
p, _ = C([{"type": "rows_removed_where", "target": None, "condition": {"column": "s", "op": "in", "value": ["y", "x"]}}])
check(ns["contract_scores"](p, g, cols)["exact_match"], "in-list order ignored")

# 5 null_count on a new column accepted
c, e = C([{"type": "columns_added", "target": ["t"]},
          {"type": "column_value_relation", "target": "t", "fn": "mul", "args": [{"col": "a"}, {"lit": 2}]}],
         [{"type": "null_count", "target": "t", "op": "eq", "value": 1}])
check(not e and not [h for h in ns["contract_issues"](c, df)[0] if "NEW column" in h], "null_count on new column allowed")

# 6 cast to string of a date-only datetime matches pandas astype(str)
dd = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-01-02"])})
c, _ = C([{"type": "column_value_relation", "target": "d", "fn": "cast", "args": [{"col": "d"}, {"lit": "string"}]}])
check(ns["verify"](dd, dd.assign(d=dd["d"].astype(str)), c)["verdict"] == "PASS", "datetime -> string cast")

# 9 / 10 schema
_, e = C([{"type": "columns_added", "target": "*"}])
check(bool(e), "columns_added '*' rejected")
_, e = C([{"type": "row_count", "target": None, "op": "decreased", "value": None}])
check(not e, "value null on relative op accepted")

# 11 sandbox blocklist
check(bool(ns["safety_check"]("import numpy as np\ndef transform(df):\n    np.save('x', df.values)\n    return df")), "np.save blocked")
print("\nPROBLEMS:", P)
