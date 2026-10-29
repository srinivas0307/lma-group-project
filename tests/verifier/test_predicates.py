"""Per-predicate tests: pass case, violation case, edge cases (README Phase 1)."""
import numpy as np
import pandas as pd
import pytest

from igt.contracts import ContractError, load_contract
from igt.verifier import verify
from igt.verifier.alignment import align
from igt.verifier.conditions import evaluate
from igt.verifier.values import dtype_family, value_equal

B = pd.DataFrame({"id": [1, 2, 3, 4, 5], "host": ["a", "a", "b", "c", "c"], "price": [10, 0, 30, 0, 50],
                  "rpm": [1.0, np.nan, 2.0, np.nan, 3.0], "room": ["x", "y", "x", "x", "y"]})


def V(preds, after, before=B, preserve=()):
    v, info = verify({"must_change": preds, "must_preserve": list(preserve)}, before, after)
    return v.status, v, info


# ---------------- values, conditions ----------------
def test_value_equality_rules():
    assert value_equal(3, 3.0) and value_equal(np.nan, None) and value_equal(pd.NaT, np.nan)
    assert value_equal(0.1 + 0.2, 0.3) and not value_equal(1, 1.1)
    assert not value_equal(True, 1) and not value_equal("1", 1) and not value_equal(None, 0)
    assert value_equal(pd.Timestamp("2020-01-01"), np.datetime64("2020-01-01"))


def test_dtype_families():
    assert dtype_family(pd.Series([1, 2])) == "int" and dtype_family(pd.Series([1, None], dtype="Int64")) == "int"
    assert dtype_family(pd.Series(["a", None])) == "string" and dtype_family(pd.Series([1.0])) == "float"
    assert dtype_family(pd.Series(["a"], dtype="category")) == "category"
    assert dtype_family(pd.to_datetime(pd.Series(["2020-01-01"]))) == "datetime"


def test_null_makes_comparisons_false():
    c = load_contract({"must_change": [{"type": "rows_removed_where", "condition": {"column": "rpm", "op": "!=", "value": 1.0}}]})
    m = evaluate(c.must_change[0].condition, B)
    assert list(m) == [False, False, True, False, True]          # nulls are false even for !=
    c2 = load_contract({"must_change": [{"type": "rows_removed_where",
                                          "condition": {"not": {"column": "rpm", "op": "==", "value": 1.0}}}]})
    assert list(evaluate(c2.must_change[0].condition, B)) == [False, True, True, True, True]


# ---------------- row-level ----------------
def test_row_count():
    assert V([{"type": "row_count", "op": "decreased"}], B.iloc[:3])[0] == "PASS"
    assert V([{"type": "row_count", "op": "unchanged"}], B.iloc[:3])[0] == "VIOLATION"
    assert V([{"type": "row_count", "op": "eq", "value": 0}], B.iloc[:0])[0] == "PASS"


def test_rows_removed_and_retained():
    cond = {"column": "price", "op": "==", "value": 0}
    keep = {"column": "price", "op": "!=", "value": 0}
    good = B[B.price != 0]
    preds = [{"type": "rows_removed_where", "condition": cond}]
    pres = [{"type": "rows_retained_where", "condition": keep}]
    assert V(preds, good, preserve=pres)[0] == "PASS"
    st, v, _ = V(preds, B, preserve=pres)
    assert st == "VIOLATION" and v.evidence_rows == [1, 3]
    assert V(preds, good.iloc[:1], preserve=pres)[0] == "VIOLATION"     # removed too much
    st, v, _ = V([{"type": "rows_removed_where", "condition": {"column": "price", "op": "==", "value": 999}}], good)
    assert st == "UNCERTAIN" and v.failures[0]["reason"] == "predicate_not_evaluable"   # matches 0 rows


def test_duplicate_count_keep_first_last():
    pred = lambda keep: [{"type": "duplicate_count", "target": ["host"], "op": "eq", "value": 0, "keep": keep}]
    assert V(pred("first"), B.drop_duplicates("host", keep="first"))[0] == "PASS"
    assert V(pred("first"), B.drop_duplicates("host", keep="last"))[0] == "VIOLATION"
    assert V(pred("last"), B.drop_duplicates("host", keep="last"))[0] == "PASS"
    assert V(pred("first"), B.drop_duplicates("host", keep=False))[0] == "VIOLATION"   # removes all copies
    assert V(pred("first"), B)[0] == "VIOLATION"                                       # duplicates left
    sorted_first = B.sort_values("price", ascending=False).drop_duplicates("host", keep="first")
    assert V(pred("first"), sorted_first)[0] == "VIOLATION"                            # sort before keep
    assert V(pred("first"), B.drop_duplicates("host").reset_index(drop=True),
             preserve=[{"type": "columns_unchanged", "target": "*", "except": []}])[0] == "PASS"


def test_duplicate_key_star_except_and_nulls_equal():
    b = pd.DataFrame({"a": [1, 1, None, None], "w": [5, 6, 7, 8]})
    pred = [{"type": "duplicate_count", "target": "*", "except": ["w"], "op": "eq", "value": 0, "keep": "first"}]
    assert V(pred, b.drop_duplicates(subset=["a"]), before=b)[0] == "PASS"
    assert V(pred, b, before=b)[0] == "VIOLATION"


# ---------------- column-level ----------------
def test_null_count():
    filled = B.assign(rpm=B.rpm.fillna(0))
    assert V([{"type": "null_count", "target": "rpm", "op": "eq", "value": 0}], filled)[0] == "PASS"
    assert V([{"type": "null_count", "target": "rpm", "op": "unchanged"}], filled)[0] == "VIOLATION"
    st, v, _ = V([{"type": "null_count", "target": "rpm", "op": "eq", "value": 0}], B.drop(columns="rpm"))
    assert st == "UNCERTAIN"


def test_columns_unchanged_and_values_unchanged():
    p = [{"type": "columns_unchanged", "target": "*", "except": ["rpm"]}]
    assert V([], B.assign(rpm=0.0), preserve=p)[0] == "PASS"
    assert V([], B.assign(price=B.price + 1), preserve=p)[0] == "VIOLATION"
    assert V([], B.astype({"price": float}), preserve=p)[0] == "VIOLATION"     # dtype family changed
    assert V([], B.drop(columns="room"), preserve=p)[0] == "VIOLATION"         # removed column
    q = [{"type": "column_values_unchanged", "target": "price"}]
    assert V([], B.astype({"price": float}), preserve=q)[0] == "PASS"          # dtype may change
    w = [{"type": "column_values_unchanged", "target": "rpm", "where": {"column": "rpm", "op": "not_null"}}]
    assert V([], B.assign(rpm=B.rpm.fillna(0)), preserve=w)[0] == "PASS"
    assert V([], B.assign(rpm=B.rpm + 1), preserve=w)[0] == "VIOLATION"


def test_column_dtype_category_order():
    cats = ["x", "y"]
    p = [{"type": "column_dtype", "target": "room", "dtype": "category", "ordered": True, "categories": cats}]
    ok = B.assign(room=pd.Categorical(B.room, categories=cats, ordered=True))
    assert V(p, ok)[0] == "PASS"
    assert V(p, B.assign(room=pd.Categorical(B.room, categories=["y", "x"], ordered=True)))[0] == "VIOLATION"
    assert V(p, B.assign(room=pd.Categorical(B.room)))[0] == "VIOLATION"       # not ordered
    assert V([{"type": "column_dtype", "target": "room", "dtype": "category"}], B)[0] == "VIOLATION"


@pytest.mark.parametrize("fn, args, after_col, ok", [
    ("sub", [{"col": "price"}, {"col": "id"}], lambda b: b.price - b.id, True),
    ("div", [{"col": "price"}, {"lit": 10}], lambda b: b.price / 10, True),
    ("div", [{"col": "price"}, {"lit": 7}], lambda b: (b.price / 7).round(), False),
    ("const", [{"lit": 0}], lambda b: 0, True),
    ("clip", [{"col": "price"}, {"lit": None}, {"lit": 20}], lambda b: b.price.clip(upper=20), True),
    ("map", [{"col": "room"}, {"lit": {"x": "X"}}], lambda b: b.room.replace({"x": "X"}), True),
    ("upper", [{"col": "room"}], lambda b: b.room.str.upper(), True),
    ("identity", [{"col": "host"}], lambda b: b.room, False),
    ("cast", [{"col": "id"}, {"lit": "string"}], lambda b: b.id.astype(str), True),
])
def test_column_value_relation(fn, args, after_col, ok):
    p = [{"type": "column_value_relation", "target": "new", "fn": fn, "args": args}]
    assert V(p, B.assign(new=after_col(B)))[0] == ("PASS" if ok else "VIOLATION")


def test_relation_where_and_missing_target():
    p = [{"type": "column_value_relation", "target": "rpm", "fn": "const", "args": [{"lit": 0}],
          "where": {"column": "rpm", "op": "is_null"}}]
    assert V(p, B.assign(rpm=B.rpm.fillna(0)))[0] == "PASS"
    assert V(p, B.assign(rpm=B.rpm.fillna(-1)))[0] == "VIOLATION"
    assert V(p, B.drop(columns="rpm"))[0] == "UNCERTAIN"


def test_columns_added_removed():
    assert V([{"type": "columns_added", "target": ["z"]}], B.assign(z=1))[0] == "PASS"
    assert V([{"type": "columns_added", "target": ["z"]}], B)[0] == "VIOLATION"
    assert V([{"type": "columns_removed", "target": ["rpm"]}], B.drop(columns="rpm"))[0] == "PASS"
    assert V([{"type": "columns_removed", "target": ["rpm"]}], B)[0] == "VIOLATION"


# ---------------- verifier-level edge cases ----------------
def test_unknown_column_is_uncertain_and_violation_wins():
    st, v, _ = V([{"type": "null_count", "target": "nope", "op": "eq", "value": 0}], B)
    assert st == "UNCERTAIN" and v.failures[0]["reason"] == "unknown_column"
    st, v, _ = V([{"type": "null_count", "target": "nope", "op": "eq", "value": 0},
                  {"type": "row_count", "op": "decreased"}], B)
    assert st == "VIOLATION" and len(v.failures) == 2


def test_invalid_contract_is_uncertain():
    v, _ = verify({"must_change": [{"type": "frobnicate"}]}, B, B)
    assert v.status == "UNCERTAIN" and v.failures[0]["reason"] == "invalid_contract"
    with pytest.raises(ContractError):
        load_contract({"must_change": [{"type": "row_count", "op": "eq"}]})


def test_empty_output_and_all_null_column():
    st, *_ = V([{"type": "row_count", "op": "decreased"}],
               B.iloc[:0], preserve=[{"type": "columns_unchanged", "target": "*", "except": []}])
    assert st == "PASS"
    allnull = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]})
    assert V([{"type": "null_count", "target": "a", "op": "eq", "value": 0}], allnull.fillna(0), before=allnull)[0] == "PASS"


def test_alignment_index_and_content():
    al = align(B, B.iloc[[4, 0, 2]], list(B.columns))
    assert al.method == "index" and al.before_ids == [4, 0, 2] and al.removed == [1, 3]
    al = align(B, B.iloc[[4, 0, 2]].reset_index(drop=True), list(B.columns))
    assert al.method == "content" and sorted(al.before_ids) == [0, 2, 4] and al.ok()


def test_reset_index_after_removing_late_rows_is_not_trusted():
    """Regression (taxi_filt_004): removing rows near the end, then reset_index(), leaves most rows in place;
    the shifted tail must not be matched by position."""
    b = pd.DataFrame({"k": range(100), "v": [i * 10 for i in range(100)]})
    out = b.drop(index=[95, 97]).reset_index(drop=True)
    al = align(b, out, ["k", "v"])
    assert al.method == "content" and sorted(al.removed) == [95, 97]


def test_ambiguous_rows_are_uncertain():
    b = pd.DataFrame({"k": [1, 1, 2], "v": [10, 20, 30]})
    after = pd.DataFrame({"k": [1, 2], "v": [99, 30]})          # index lost; which k=1 row survived?
    p = [{"type": "column_values_unchanged", "target": "k"}]
    v, info = verify({"must_preserve": p}, b, after)
    assert v.status == "UNCERTAIN" and v.failures[0]["reason"] == "rows_unmatched"


def test_directions_filter():
    c = {"must_change": [{"type": "row_count", "op": "decreased"}],
         "must_preserve": [{"type": "columns_unchanged", "target": "*", "except": []}]}
    damaged = B.iloc[:3].assign(price=0)
    assert verify(c, B, damaged)[0].status == "VIOLATION"
    assert verify(c, B, damaged, ("must_change",))[0].status == "PASS"
    assert verify(c, B, damaged, ("must_preserve",))[0].status == "VIOLATION"
