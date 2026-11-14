"""Answer key: is a candidate output equal to the reference output? (README Phase 2, `outputs_equal`)

Same rules as the Baseline 1 Kaggle harness, so labels are identical:
- compares content; row order, index and column order are ignored
- same column set, same row count, same dtype family per column, ordered categories in the same order
- values: floats rounded to 9 decimals; missing values (None/NaN/NaT/NA) are equal to each other
"""
import numpy as np
import pandas as pd


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


def norm(v):
    if v is None:
        return ("N",)
    try:
        if pd.isna(v):
            return ("N",)
    except (TypeError, ValueError):
        pass
    if isinstance(v, (bool, np.bool_)):
        return ("B", bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return ("X", round(float(v), 9))
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return ("T", pd.Timestamp(v).isoformat())
    return ("S", str(v))


def outputs_equal(ref, cand):
    """Returns (equal: bool, reason: str)."""
    if not isinstance(cand, pd.DataFrame):
        return False, "not a DataFrame"
    if cand.columns.duplicated().any():
        return False, "duplicate column names"
    if set(ref.columns) != set(cand.columns):
        extra = sorted(map(str, set(cand.columns) - set(ref.columns)))
        missing = sorted(map(str, set(ref.columns) - set(cand.columns)))
        return False, f"columns differ (extra={extra}, missing={missing})"
    if len(ref) != len(cand):
        return False, f"row count differs ({len(cand)} vs {len(ref)})"
    cols = list(ref.columns)
    for c in cols:
        fr, fc = family(ref[c]), family(cand[c])
        if fr != fc:
            return False, f"dtype family differs in {c} ({fc} vs {fr})"
        if fr == "category" and ref[c].cat.ordered:
            if not cand[c].cat.ordered or list(ref[c].cat.categories) != list(cand[c].cat.categories):
                return False, f"category order differs in {c}"

    def rows(df):
        return sorted(zip(*[[norm(v) for v in df[c].astype(object).tolist()] for c in cols]))

    if rows(ref) != rows(cand):
        return False, "values differ"
    return True, "equal"
