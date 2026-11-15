# ===== Before/after profiler + deterministic intent verifier =====
# Verdict: PASS (every predicate holds) | VIOLATION (some predicate fails -> repair)
#          | UNCERTAIN (the contract itself cannot be checked on this data -> flag, no repair).
# Deviation from the written spec, on purpose: problems caused by the OUTPUT (a required column
# missing, a renumbered index) are VIOLATIONS, because the repair agent can fix them; only problems
# of the CONTRACT (unknown input column, removal condition matching no rows) are UNCERTAIN.

PASS, FAIL, UNCERTAIN = "PASS", "FAIL", "UNCERTAIN"
MAX_EXAMPLES = 5


def _fmt(v):
    if is_null(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if _is_datetime(v):
        return str(pd.Timestamp(v))
    return v if isinstance(v, (str, int, float, bool)) else str(v)


def _count_ok(op, before_n, after_n, value):
    return {"unchanged": after_n == before_n, "decreased": after_n < before_n, "increased": after_n > before_n,
            "eq": after_n == value, "lt": after_n < (value if value is not None else 0), "le": after_n <= (value or 0),
            "gt": after_n > (value or 0), "ge": after_n >= (value or 0)}[op]


# ---------------------------------------------------------------- relation functions
def _arg_series(arg, before_rows):
    if "col" in arg:
        return before_rows[arg["col"]].astype(object) if isinstance(before_rows[arg["col"]].dtype, pd.CategoricalDtype) else before_rows[arg["col"]]
    return pd.Series([arg["lit"]] * len(before_rows), index=before_rows.index, dtype=object)


def _cast_value(v, dtype, fmt=None):
    if is_null(v):
        return None
    try:
        if dtype == "int":
            f = float(v)
            return int(f) if f.is_integer() else None
        if dtype == "float":
            return float(v)
        if dtype == "string":
            if isinstance(v, str):
                return v
            if _is_datetime(v):     # like pandas astype(str): date only when there is no time part
                ts = pd.Timestamp(v)
                return str(ts.date()) if ts == ts.normalize() else str(ts)
            return str(_fmt(v))
        if dtype == "bool":
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if _is_number(v):
                return {0: False, 1: True}.get(float(v), None)
            return {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}.get(str(v).strip().lower())
        if dtype == "category":
            return v
        if dtype == "datetime":
            if _is_datetime(v):
                return pd.Timestamp(v)
            return pd.to_datetime(str(v), format=fmt) if fmt else pd.to_datetime(str(v), format="ISO8601")
    except Exception:
        return None
    return None


def relation_expected(p, before_rows):
    """Expected after-values for column_value_relation, computed from the BEFORE rows."""
    fn, args = p["fn"], p.get("args", [])
    idx = before_rows.index
    if fn == "null":
        return pd.Series([None] * len(idx), index=idx, dtype=object)
    if fn in ("identity", "const"):
        return _arg_series(args[0], before_rows)
    if fn in ("add", "sub", "mul", "div"):
        a, b = _arg_series(args[0], before_rows), _arg_series(args[1], before_rows)
        out = []
        for x, y in zip(a.tolist(), b.tolist()):
            if is_null(x) or is_null(y):
                out.append(None); continue
            try:
                x, y = float(x), float(y)
                out.append({"add": x + y, "sub": x - y, "mul": x * y}.get(fn) if fn != "div" else (None if y == 0 else x / y))
            except Exception:
                out.append(None)
        return pd.Series(out, index=idx, dtype=object)
    src = _arg_series(args[0], before_rows)
    vals = src.tolist()
    if fn == "round":
        n = args[1]["lit"]
        return pd.Series([None if is_null(x) else float(np.round(float(x), n)) for x in vals], index=idx, dtype=object)
    if fn == "clip":
        lo, hi = args[1]["lit"], args[2]["lit"]
        def c(x):
            if is_null(x): return None
            x = float(x)
            if lo is not None: x = max(x, float(lo))
            if hi is not None: x = min(x, float(hi))
            return x
        return pd.Series([c(x) for x in vals], index=idx, dtype=object)
    if fn in ("lower", "upper", "strip"):
        return pd.Series([getattr(x, fn)() if isinstance(x, str) else (None if is_null(x) else x) for x in vals], index=idx, dtype=object)
    if fn == "map":
        m = args[1]["lit"]
        def mp(x):
            if is_null(x): return None
            try:
                if x in m: return m[x]
            except TypeError:
                pass
            key = str(_fmt(x))
            if key in m: return m[key]
            if isinstance(x, (float, np.floating)) and float(x).is_integer() and str(int(x)) in m:
                return m[str(int(x))]
            return x
        return pd.Series([mp(x) for x in vals], index=idx, dtype=object)
    if fn == "cast":
        dtype = args[1]["lit"]
        fmt = args[2]["lit"] if len(args) > 2 else None
        return pd.Series([_cast_value(x, dtype, fmt) for x in vals], index=idx, dtype=object)
    raise ValueError(f"unknown fn {fn}")


# ---------------------------------------------------------------- predicate checks
def _value_mismatch_examples(before, after, rows, col, expected=None, atol=1e-9):
    """Rows (among `rows`) where after[col] != expected (default: before[col])."""
    exp = expected if expected is not None else before.loc[rows, col]
    got = after.loc[rows, col]
    ok = series_equal_mask(got, exp.reindex(rows), atol)
    bad = list(ok.index[~ok.values])
    ex = [{"row": _fmt(r), "before": _fmt(before.at[r, col]) if col in before.columns else None,
           "after": _fmt(got.at[r]), **({"expected": _fmt(exp.at[r])} if expected is not None else {})}
          for r in bad[:MAX_EXAMPLES]]
    return len(bad), ex


def check_predicate(p, before, after, surv):
    """Returns (status, detail, examples)."""
    t = p["type"]
    bcols, acols = set(before.columns), set(after.columns)
    try:
        if t == "row_count":
            ok = _count_ok(p["op"], len(before), len(after), p.get("value"))
            return (PASS if ok else FAIL), f"rows before={len(before)}, after={len(after)}, required op={p['op']}" + (f" {p.get('value')}" if "value" in p else ""), []

        if t in ("rows_removed_where", "rows_retained_where"):
            m = eval_condition(before, p["condition"])
            matched = before.index[m.values]
            if t == "rows_removed_where":
                if len(matched) == 0:
                    return UNCERTAIN, "the removal condition matches no input row", []
                kept = [r for r in matched if r in surv]
                ex = [{"row": _fmt(r), **{c: _fmt(before.at[r, c]) for c in sorted(condition_columns(p["condition"]))}} for r in kept[:MAX_EXAMPLES]]
                return (PASS if not kept else FAIL), f"{len(matched)} input rows match the condition; {len(kept)} of them are still present (all must be removed)", ex
            lost = [r for r in matched if r not in surv]
            ex = [{"row": _fmt(r), **{c: _fmt(before.at[r, c]) for c in sorted(condition_columns(p["condition"]))}} for r in lost[:MAX_EXAMPLES]]
            return (PASS if not lost else FAIL), f"{len(matched)} input rows match the condition; {len(lost)} of them were removed (all must be kept)", ex

        if t == "duplicate_count":
            keys = expand_targets(p, list(before.columns))
            miss = [k for k in keys if k not in acols]
            if miss:
                return FAIL, f"key columns missing from the output: {miss}", []
            n_dup = int(after.duplicated(subset=keys).sum())
            ok = _count_ok(p["op"], int(before.duplicated(subset=keys).sum()), n_dup, p.get("value"))
            detail = f"duplicates on {keys} in output: {n_dup}"
            ex = []
            if ok and "keep" in p:
                expected = set(before.index[~before.duplicated(subset=keys, keep=p["keep"]).values])
                got = set(surv)
                missing = sorted(expected - got)
                wrong = sorted(got - expected)
                if missing or wrong:
                    ok = False
                    detail += f"; with keep={p['keep']}: {len(missing)} rows that should be the surviving {p['keep']} of their group are missing, {len(wrong)} other rows survive"
                    ex = [{"row": _fmt(r), "problem": f"should survive (it is the {p['keep']} of its key group)"} for r in missing[:3]] + \
                         [{"row": _fmt(r), "problem": "should be removed (not the kept row of its key group)"} for r in wrong[:2]]
            return (PASS if ok else FAIL), detail, ex

        if t == "null_count":
            col = p["target"]
            if col not in acols:
                return FAIL, f"column {col!r} is missing from the output", []
            b = int(before[col].isna().sum()) if col in bcols else 0
            a = int(after[col].isna().sum())
            ok = _count_ok(p["op"], b, a, p.get("value"))
            return (PASS if ok else FAIL), f"nulls in {col!r}: before={b}, after={a}, required op={p['op']}" + (f" {p.get('value')}" if "value" in p else ""), []

        if t in ("column_values_unchanged", "columns_unchanged"):
            cols = expand_targets(p, list(before.columns))
            rows = list(surv)
            if p.get("where"):
                wm = eval_condition(before, p["where"])
                wset = set(before.index[wm.values])
                rows = [r for r in rows if r in wset]
            problems, ex = [], []
            for col in cols:
                if col not in bcols:
                    return UNCERTAIN, f"column {col!r} is not in the input data", []
                if col not in acols:
                    problems.append(f"column {col!r} is missing from the output"); continue
                if t == "columns_unchanged":
                    fb, fa = dtype_family(before[col]), dtype_family(after[col])
                    if fb != fa:
                        problems.append(f"column {col!r} changed dtype family {fb} -> {fa}")
                        continue
                n_bad, e = _value_mismatch_examples(before, after, rows, col)
                if n_bad:
                    problems.append(f"column {col!r}: {n_bad} surviving rows changed value")
                    ex += [{"column": col, **x} for x in e[:3]]
            if problems:
                return FAIL, "; ".join(problems[:6]), ex[:MAX_EXAMPLES]
            return PASS, f"{len(cols)} column(s) unchanged on {len(rows)} surviving rows", []

        if t == "column_dtype":
            col = p["target"]
            if col not in acols:
                return FAIL, f"column {col!r} is missing from the output", []
            fam = dtype_family(after[col])
            if fam != p["dtype"]:
                return FAIL, f"column {col!r} has dtype {after[col].dtype} (family {fam}), required {p['dtype']}", []
            if p["dtype"] == "category":
                cat = after[col].cat
                want_ordered = bool(p.get("ordered", False))   # omitted = unordered: no ordering the request did not ask for
                if bool(cat.ordered) != want_ordered:
                    return FAIL, (f"column {col!r} is an ordered category, but the contract does not ask for an order "
                                  f"(use astype('category'))" if cat.ordered else
                                  f"column {col!r} ordered={bool(cat.ordered)}, required ordered={want_ordered}"), []
                if "categories" in p:
                    got = [_fmt(c) for c in cat.categories]
                    want = p["categories"]
                    if p.get("ordered") or cat.ordered:
                        same = len(got) == len(want) and all(scalar_equal(a, b) for a, b in zip(got, want))
                    else:
                        same = len(got) == len(want) and all(any(scalar_equal(a, b) for b in want) for a in got)
                    if not same:
                        return FAIL, f"column {col!r} categories {got}, required {want}", []
            return PASS, f"column {col!r} dtype {after[col].dtype}", []

        if t == "column_value_relation":
            col = p["target"]
            for a in p.get("args", []):
                if "col" in a and a["col"] not in bcols:
                    return UNCERTAIN, f"argument column {a['col']!r} is not in the input data", []
            if col not in acols:
                return FAIL, f"column {col!r} is missing from the output", []
            rows = list(surv)
            if p.get("where"):
                wm = eval_condition(before, p["where"])
                wset = set(before.index[wm.values])
                rows = [r for r in rows if r in wset]
            if not rows:
                return PASS, "no surviving rows match the where-condition", []
            exp = relation_expected(p, before.loc[rows])
            n_bad, ex = _value_mismatch_examples(before, after, rows, col, expected=exp, atol=float(p.get("atol", 1e-9)))
            arg_cols = [a["col"] for a in p.get("args", []) if "col" in a]
            for e in ex:   # show the input values the expected value is computed from (the target may be a new column)
                if col not in bcols:
                    e.pop("before", None)
                e["inputs"] = {c: _fmt(before.at[e["row"], c]) for c in arg_cols}
            if n_bad:
                return FAIL, f"{col!r} != {p['fn']}(...) on {n_bad} of {len(rows)} checked rows", ex
            return PASS, f"{col!r} == {p['fn']}(...) on {len(rows)} rows", []

        if t == "columns_added":
            bad = [c for c in p["target"] if c in bcols or c not in acols]
            return (PASS if not bad else FAIL), (f"columns not newly added: {bad}" if bad else f"added {p['target']}"), []

        if t == "columns_removed":
            bad = [c for c in p["target"] if c not in bcols or c in acols]
            return (PASS if not bad else FAIL), (f"columns not removed: {bad}" if bad else f"removed {p['target']}"), []
    except MissingColumn as e:
        return UNCERTAIN, f"condition column {e.column!r} is not in the input data", []
    except Exception as e:   # e.g. round/clip on a text column: the contract cannot be evaluated on this data
        return UNCERTAIN, f"predicate cannot be evaluated on this data ({type(e).__name__}: {str(e)[:150]})", []
    return UNCERTAIN, f"unsupported predicate {t}", []


def profile_change(before, after):
    """Compact before/after profile used in reports and by the repair agent."""
    if not isinstance(after, pd.DataFrame):
        return {"error": "output is not a DataFrame"}
    if isinstance(after.columns, pd.MultiIndex) or after.columns.duplicated().any():   # after[c] would be a DataFrame
        return {"multilevel_columns": isinstance(after.columns, pd.MultiIndex),
                "rows_before": len(before), "rows_after": len(after),
                "duplicate_columns": sorted(set(after.columns[after.columns.duplicated()]))}
    surv = after.index.intersection(before.index)
    changed = {}
    for c in before.columns:
        if c in after.columns and len(surv):
            try:
                n = int((~series_equal_mask(after.loc[surv, c], before.loc[surv, c])).sum())
            except Exception:
                n = -1
            if n:
                changed[c] = n
    return {"rows_before": len(before), "rows_after": len(after),
            "rows_removed": int(len(before.index.difference(after.index))),
            "columns_added": [c for c in after.columns if c not in before.columns],
            "columns_removed": [c for c in before.columns if c not in after.columns],
            "dtype_changes": {c: f"{before[c].dtype} -> {after[c].dtype}" for c in before.columns
                              if c in after.columns and str(before[c].dtype) != str(after[c].dtype)},
            "cells_changed_by_column": changed}


def verify(before, after, contract):
    results, structural = [], []
    if not isinstance(after, pd.DataFrame):
        return {"verdict": "VIOLATION", "structural": ["the transformation did not return a DataFrame"], "predicates": []}
    if isinstance(after.columns, pd.MultiIndex):
        return {"verdict": "VIOLATION", "predicates": [], "profile": profile_change(before, after),
                "structural": ["the output has multi-level column names (e.g. from groupby().agg); return a frame with the "
                               "original flat column names"]}
    if after.columns.duplicated().any():
        structural.append(f"duplicate column names in the output: {list(after.columns[after.columns.duplicated()])}")
    if after.index.has_duplicates or not after.index.isin(before.index).all():
        structural.append("the output index does not keep the original row ids (groupby/merge/reset_index/ignore_index "
                          "rebuild the index); keep the index labels of the input rows, e.g. use drop_duplicates or a boolean mask on df")
    elif len(after) < len(before) and after.index.equals(pd.RangeIndex(len(after))):
        shared = [c for c in before.columns if c in after.columns]
        if shared:
            same = pd.Series(True, index=after.index)
            for c in shared:
                same &= series_equal_mask(after[c], before.loc[after.index, c]).values
            if same.mean() < 0.5:
                structural.append(f"the output index was renumbered 0..{len(after) - 1} (reset_index / groupby / merge / ignore_index), "
                                  f"so {int((~same).sum())} rows no longer carry their original row ids; do not renumber the index, "
                                  f"e.g. use df.drop_duplicates(...) or df.loc[mask] directly")
    declared_new = set()
    for sec in ("must_change", "must_preserve"):
        for p in contract[sec]:
            if p["type"] == "columns_added":
                declared_new |= set(p["target"])
            if p["type"] in ("column_value_relation", "column_dtype"):
                declared_new.add(p["target"])
    extra = [c for c in after.columns if c not in before.columns and c not in declared_new]
    if extra:
        structural.append(f"the output has columns the contract does not ask for: {extra}")
    if structural:
        return {"verdict": "VIOLATION", "structural": structural, "predicates": [], "profile": profile_change(before, after)}
    surv = set(after.index)
    for sec in ("must_change", "must_preserve"):
        for i, p in enumerate(contract[sec]):
            st, detail, ex = check_predicate(p, before, after, surv)
            results.append({"section": sec, "index": i, "type": p["type"], "status": st, "detail": detail,
                            "examples": ex, "predicate": p})
    stats = [r["status"] for r in results]
    verdict = "VIOLATION" if FAIL in stats else ("UNCERTAIN" if UNCERTAIN in stats else "PASS")
    return {"verdict": verdict, "structural": [], "predicates": results, "profile": profile_change(before, after)}


def render_report(rep, max_items=8):
    """Human/LLM-readable verifier report (failed and uncertain predicates only)."""
    lines = [f"Verifier verdict: {rep['verdict']}"]
    for s in rep.get("structural", []):
        lines.append(f"- STRUCTURE: {s}")
    bad = [r for r in rep.get("predicates", []) if r["status"] != PASS]
    for r in bad[:max_items]:
        lines.append(f"- {r['status']} {r['section']}[{r['index']}] {json.dumps(r['predicate'], default=str)}")
        lines.append(f"    {r['detail']}")
        for e in r["examples"][:3]:
            lines.append(f"    example: {json.dumps(e, default=str)}")
    if rep.get("profile") and "error" not in rep["profile"]:
        lines.append(f"Profile of your output vs input: {json.dumps(rep['profile'], default=str)}")
    return "\n".join(lines)
