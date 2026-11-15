"""Crash fuzzing: mutated contracts and odd outputs must never raise inside schema / checks / verifier / evaluator."""
import sys, json, copy, random, traceback, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import harness
import pandas as pd, numpy as np
ns = harness.load()
src = (harness.HERE / "cells" / "c09_parser_selective_facts.py").read_text()
exec(src.replace("__SYSTEM_PROMPT__", "''"), ns)
rng = random.Random(0)
crashes = collections.Counter(); examples = {}

def mutate(c, cols):
    c = copy.deepcopy(c)
    preds = c["must_change"] + c["must_preserve"]
    if not preds: return c
    p = rng.choice(preds)
    k = rng.randrange(14)
    if k == 0 and "condition" in p: p["condition"] = {"column": rng.choice(cols), "op": rng.choice(["<", "==", "in", "not_in", ">="]), "value": rng.choice([0, "x", 1.5, ["a", 1], [None], True, "2020-01-01"])}
    elif k == 1: p["target"] = rng.choice(cols + ["*", ["zz"], [cols[0], cols[-1]], None])
    elif k == 2: p.update(type="column_value_relation", target=rng.choice(cols), fn=rng.choice(["round", "clip", "div", "lower", "map", "cast", "add"]),
                          args=[{"col": rng.choice(cols)}] + rng.choice([[{"lit": 1}], [{"lit": None}, {"lit": 3}], [{"lit": {"a": "b", "1": 2}}], [{"lit": "datetime"}], [{"lit": "int"}], [{"lit": "bool"}], [{"col": rng.choice(cols)}]]))
    elif k == 3: p.update(type="column_dtype", target=rng.choice(cols), dtype="category", ordered=True, categories=rng.choice([[1, 2], ["x"], [None], [1.5, "a"]]))
    elif k == 4: p.update(type="duplicate_count", target=rng.choice(["*", [rng.choice(cols)], [cols[0], cols[1]]]), op="eq", value=0, keep=rng.choice(["first", "last"]))
    elif k == 5: p.update(type="null_count", target=rng.choice(cols + ["new_col"]), op=rng.choice(["eq", "unchanged", "increased"]), value=0)
    elif k == 6: p["where"] = {"not": {"any": [{"column": rng.choice(cols), "op": "is_null"}, {"column": rng.choice(cols), "op": "!=", "value": rng.choice([0, "y", 2.5])}]}}
    elif k == 7: p.update(type="columns_added", target=[rng.choice(cols + ["new_col"])])
    elif k == 8: p.update(type="columns_removed", target=[rng.choice(cols + ["ghost"])])
    elif k == 9: p.update(type="column_values_unchanged", target=rng.choice(cols + ["*"]), where={"column": rng.choice(cols), "op": "in", "value": [1, "a", None]})
    elif k == 10: p.update(type="rows_retained_where", target=None, condition={"all": [{"column": rng.choice(cols), "op": ">", "value": "abc"}]})
    elif k == 11: p.update(type="column_value_relation", target="new_col", fn="cast", args=[{"col": rng.choice(cols)}, {"lit": "datetime"}, {"lit": "%d/%m/%Y"}])
    elif k == 12: p.update(type="row_count", target=None, op=rng.choice(["eq", "lt", "ge"]), value=rng.randrange(0, 2000))
    else: c["must_change"] = []
    return c

def odd_outputs(before, ref):
    yield ref.iloc[0:0]                                   # empty
    yield ref[[]]                                         # no columns
    yield ref.rename(columns={ref.columns[0]: 0})          # int column name
    o = ref.copy(); o.index = o.index.astype(str); yield o # string index
    o = ref.copy(); o.columns = pd.MultiIndex.from_tuples([(c, "x") for c in o.columns]); yield o
    o = ref.copy(); o[o.columns[0]] = pd.to_datetime("2020-01-01").tz_localize("UTC"); yield o
    o = ref.copy(); o[o.columns[1]] = [[1, 2]] * len(o); yield o   # list cells (unhashable)
    o = ref.copy(); o[o.columns[1]] = [{"a": 1}] * len(o); yield o # dict cells
    o = ref.copy(); o[o.columns[0]] = pd.Categorical(o[o.columns[0]].astype(str)); yield o
    o = ref.copy(); o[o.columns[0]] = o[o.columns[0]].astype("string") if o[o.columns[0]].dtype == object else o[o.columns[0]].astype("Float64"); yield o
    o = ref.copy(); o[o.columns[0]] = np.inf; yield o
    yield pd.concat([ref, ref.iloc[:2]])                  # duplicated index

def run(label, fn):
    try:
        fn()
    except Exception as e:
        key = f"{label}: {type(e).__name__}: {str(e)[:90]}"
        crashes[key] += 1
        examples.setdefault(key, traceback.format_exc().strip().splitlines()[-3:])

for t in ns["ALL_TASKS"]:
    before = ns["load_before"](t); ref = ns["reference_output"](t); cols = list(before.columns)
    gold, _ = ns["normalize_contract"](t["gold_contract"])
    for _ in range(25):
        m = mutate(t["gold_contract"], cols)
        def chain(m=m):
            c, errs = ns["normalize_contract"](m)
            if errs: return
            ns["contract_issues"](c, before)
            ns["verify"](before, ref, c)
            ns["render_report"](ns["verify"](before, before.copy(), c))
            ns["contract_scores"](c, gold, cols)
            ns["facts_issues"](c, [{"value": 1.0, "scope": {"kind": "full_column"}}])
        run("contract-chain", chain)
    for o in odd_outputs(before, ref):
        run("verify-odd-output", lambda o=o: ns["render_report"](ns["verify"](before, o, gold)))
        run("outputs_match-odd", lambda o=o: ns["outputs_match"](o, ref))
        run("profile-odd", lambda o=o: ns["profile_change"](before, o))
    run("compute_facts", lambda: ns["compute_facts"](t["request"], before) if True else None)
print("tasks:", len(ns["ALL_TASKS"]))
for k, v in crashes.most_common():
    if k.startswith("compute_facts") and "FactUnavailable" in k: continue
    print(f"{v:4d}x {k}\n       " + "\n       ".join(examples[k]))
