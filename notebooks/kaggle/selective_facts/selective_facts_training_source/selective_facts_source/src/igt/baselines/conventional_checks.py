"""Baseline 2: conventional data-quality checks (verifier slot = "conventional").

The checks see only the input DataFrame and the candidate's output. They never see the request,
the contract or the reference. That is exactly what the research question compares against.

Two steps, so thresholds can be tuned cheaply on the dev split:
    measure(before, after) -> dict of raw measurements (computed once per candidate)
    judge(measurements, thresholds) -> Verdict dict (status PASS / VIOLATION, failures, evidence_rows)

Checks (README §4.5): schema, nulls, dtypes, uniqueness, ranges, row consistency, distribution shift.
"""
import math

import numpy as np
import pandas as pd

INF = float("inf")
NUMERIC = {"int", "float"}
CATEGORICAL = {"string", "object", "category", "bool"}
# dtype changes that keep the values' meaning (used when dtype_mode = "compatible")
COMPATIBLE = {frozenset(p) for p in [("int", "float"), ("string", "category"), ("object", "category"),
                                     ("string", "object"), ("bool", "int")]}
MAX_EVIDENCE = 5


def family(s):
    dt = s.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return "category"
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    if pd.api.types.is_integer_dtype(dt):
        return "int"
    if pd.api.types.is_float_dtype(dt):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dt):
        return "datetime"
    if pd.api.types.is_object_dtype(dt):
        return "string" if all(isinstance(v, str) for v in s.dropna()) else "object"
    if pd.api.types.is_string_dtype(dt):
        return "string"
    return str(dt)


def _tvd(a, b):
    p = a.astype(str).value_counts(normalize=True, dropna=True)
    q = b.astype(str).value_counts(normalize=True, dropna=True)
    keys = p.index.union(q.index)
    return 0.5 * float(np.abs(p.reindex(keys, fill_value=0) - q.reindex(keys, fill_value=0)).sum())


def _evidence(mask_index):
    return [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in list(mask_index)[:MAX_EVIDENCE]]


def _hashable(df):
    """Copy where unhashable cells (lists, dicts, Index objects...) are replaced by their repr, so
    duplicate/uniqueness checks work on any output. Columns without such cells are left as they are."""
    out = None
    for c in df.columns[df.dtypes == object] if len(df.columns) else []:
        col = df[c] if not isinstance(df[c], pd.DataFrame) else None
        if col is None:
            continue
        bad = col.map(lambda v: not isinstance(v, (str, int, float, bool, type(None), np.generic, pd.Timestamp))
                      and getattr(v, "__hash__", None) is None)
        if bad.any():
            out = df.copy() if out is None else out
            out[c] = col.where(~bad, col.map(repr))
    return df if out is None else out


def _index_subset(before_idx, after_idx):
    """True if the output index is unique and every label is an input row label.
    A different number of index levels (e.g. a MultiIndex left by groupby().apply) is never a subset."""
    if after_idx.nlevels != before_idx.nlevels or not after_idx.is_unique:
        return False
    try:
        return bool(after_idx.isin(before_idx).all())
    except (TypeError, ValueError):
        return False


def measure(before, after):
    """Raw measurements. Each entry: value, plus the column/rows responsible when relevant."""
    after = _hashable(after)
    m = {}
    n_b, n_a = len(before), len(after)
    b_cols, a_cols = [str(c) for c in before.columns], [str(c) for c in after.columns]
    common = [c for c in before.columns if c in after.columns]
    new_cols = [c for c in after.columns if c not in before.columns]

    # schema
    m["duplicate_column_names"] = dict(value=int(pd.Index(a_cols).duplicated().sum()))
    dropped = sorted(set(b_cols) - set(a_cols))
    m["dropped_columns"] = dict(value=len(dropped), columns=dropped)
    if m["duplicate_column_names"]["value"]:
        after = after.loc[:, ~after.columns.duplicated()]
        common = [c for c in before.columns if c in after.columns]

    # rows
    m["row_ratio"] = dict(value=(n_a / n_b) if n_b else 1.0, rows_before=n_b, rows_after=n_a)
    idx_subset = _index_subset(before.index, after.index)
    m["index_not_subset"] = dict(value=int(not idx_subset))

    # nulls
    worst, col = 0.0, None
    for c in common:
        inc = (after[c].isna().mean() if n_a else 0.0) - (before[c].isna().mean() if n_b else 0.0)
        if inc > worst:
            worst, col = float(inc), str(c)
    m["null_increase"] = dict(value=worst, column=col)
    worst, col = 0.0, None
    for c in new_cols:
        f = float(after[c].isna().mean()) if n_a else 0.0
        if f > worst:
            worst, col = f, str(c)
    m["new_column_null_frac"] = dict(value=worst, column=col)

    # dtypes
    fam = {c: (family(before[c]), family(after[c])) for c in common}
    changed = [str(c) for c, (fb, fa) in fam.items() if fb != fa]
    incompatible = [str(c) for c, (fb, fa) in fam.items() if fb != fa and frozenset((fb, fa)) not in COMPATIBLE]
    m["dtype_changes"] = dict(value=len(changed), columns=changed)
    m["dtype_incompatible_changes"] = dict(value=len(incompatible), columns=incompatible)

    # uniqueness
    lost = [str(c) for c in common if n_b and before[c].notna().all() and before[c].is_unique
            and not after[c].dropna().is_unique]
    m["lost_unique_columns"] = dict(value=len(lost), columns=lost)
    dup_b = int(before.duplicated().sum())
    dup_a_mask = after.duplicated() if n_a else pd.Series(dtype=bool)
    m["duplicate_rows_increase"] = dict(value=max(0, int(dup_a_mask.sum()) - dup_b),
                                        evidence_rows=_evidence(after.index[dup_a_mask]) if n_a else [])

    # ranges and distribution shift (only columns whose dtype family didn't change)
    oor, oor_col, oor_rows = 0.0, None, []
    z, z_col = 0.0, None
    tvd, tvd_col = 0.0, None
    newv, newv_col, newv_rows = 0.0, None, []
    for c in common:
        fb, fa = fam[c]
        if fb != fa or n_a == 0:
            continue
        b, a = before[c], after[c]
        if fb in NUMERIC:
            bn, an = b.dropna().astype(float), a.dropna().astype(float)
            if len(bn) == 0 or len(an) == 0:
                continue
            lo, hi = bn.min(), bn.max()
            out_mask = (an < lo - 1e-9 * max(1, abs(lo))) | (an > hi + 1e-9 * max(1, abs(hi)))
            frac = float(out_mask.mean())
            if frac > oor:
                oor, oor_col, oor_rows = frac, str(c), _evidence(an.index[out_mask])
            sd = bn.std(ddof=0)
            dz = abs(an.mean() - bn.mean()) / sd if sd > 0 else (0.0 if an.mean() == bn.mean() else INF)
            if dz > z:
                z, z_col = float(dz), str(c)
        elif fb in CATEGORICAL:
            d = _tvd(b, a)
            if d > tvd:
                tvd, tvd_col = d, str(c)
            seen = set(b.dropna().astype(str))
            an = a.dropna().astype(str)
            if len(an):
                nv_mask = ~an.isin(seen)
                frac = float(nv_mask.mean())
                if frac > newv:
                    newv, newv_col, newv_rows = frac, str(c), _evidence(an.index[nv_mask])
    m["out_of_range_frac"] = dict(value=oor, column=oor_col, evidence_rows=oor_rows)
    m["mean_shift_z"] = dict(value=z, column=z_col)
    m["category_tvd"] = dict(value=tvd, column=tvd_col)
    m["new_value_frac"] = dict(value=newv, column=newv_col, evidence_rows=newv_rows)
    return m


# check name -> (measurement, comparison, threshold key)
#   "max": violation if value > threshold;  "min": violation if value < threshold
RULES = [
    ("schema.duplicate_column_names", "duplicate_column_names", "max", None),        # always 0
    ("schema.dropped_columns", "dropped_columns", "max", "max_dropped_columns"),
    ("rows.empty_output", "row_ratio", "min_strict", None),                            # always > 0
    ("rows.rows_added", "row_ratio", "max", "max_row_ratio"),
    ("rows.rows_removed", "row_ratio", "min", "min_row_ratio"),
    ("rows.index_not_subset", "index_not_subset", "max", "max_index_not_subset"),
    ("nulls.null_increase", "null_increase", "max", "max_null_increase"),
    ("nulls.new_column_nulls", "new_column_null_frac", "max", "max_new_column_null_frac"),
    ("dtypes.changed", None, "dtype", "dtype_mode"),
    ("uniqueness.lost_unique_columns", "lost_unique_columns", "max", "max_lost_unique_columns"),
    ("uniqueness.duplicate_rows_increase", "duplicate_rows_increase", "max", "max_duplicate_rows_increase"),
    ("ranges.out_of_range", "out_of_range_frac", "max", "max_out_of_range_frac"),
    ("distribution.mean_shift", "mean_shift_z", "max", "max_mean_shift_z"),
    ("distribution.category_shift", "category_tvd", "max", "max_category_tvd"),
    ("distribution.new_values", "new_value_frac", "max", "max_new_value_frac"),
]


def _num(x):
    return INF if x is None or (isinstance(x, str) and x.lower() in {"inf", "off", "none"}) else float(x)


def judge(m, thresholds):
    """Apply thresholds to measurements. Returns Verdict(status, failures, evidence_rows) as a dict."""
    failures, evidence = [], []
    for name, key, kind, tkey in RULES:
        if kind == "dtype":
            mode = thresholds.get("dtype_mode", "compatible")
            src = {"strict": "dtype_changes", "compatible": "dtype_incompatible_changes"}.get(mode)
            if src and m[src]["value"] > 0:
                failures.append(dict(kind="check", check=name, value=m[src]["value"], threshold=f"mode={mode}",
                                     columns=m[src]["columns"]))
            continue
        val = m[key]["value"]
        if kind == "min_strict":
            bad, thr = m[key]["rows_after"] == 0, "> 0 rows"
        elif kind == "max":
            thr = 0 if tkey is None else _num(thresholds.get(tkey))
            bad = val > thr
        else:
            thr = 0 if tkey is None else _num(thresholds.get(tkey, 0))
            bad = val < thr
        if bad:
            f = dict(kind="check", check=name, value=round(val, 6) if isinstance(val, float) and math.isfinite(val) else val,
                     threshold=thr)
            for extra in ("column", "columns"):
                if m[key].get(extra):
                    f[extra] = m[key][extra]
            failures.append(f)
            evidence += m[key].get("evidence_rows", [])
    return dict(status="VIOLATION" if failures else "PASS", failures=failures,
                evidence_rows=list(dict.fromkeys(evidence))[:MAX_EVIDENCE])
