"""Value equality and dtype families (docs/eval_set_and_contract_guide.md §5.2, §5.7)."""
import math

import numpy as np
import pandas as pd

DEFAULT_ATOL = 1e-9


def is_null(v) -> bool:
    if v is None:
        return True
    try:
        r = pd.isna(v)
        return bool(r) if np.ndim(r) == 0 else False
    except (TypeError, ValueError):
        return False


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _is_bool(v) -> bool:
    return isinstance(v, (bool, np.bool_))


def _is_time(v) -> bool:
    return isinstance(v, (pd.Timestamp, np.datetime64)) or (hasattr(v, "isoformat") and hasattr(v, "year"))


def value_equal(a, b, atol: float = DEFAULT_ATOL) -> bool:
    """Both null, or both non-null and equal. Numbers compare by value across int/float (3 == 3.0), with atol.
    Booleans only equal booleans; strings compare exactly; datetimes as timestamps."""
    na, nb = is_null(a), is_null(b)
    if na or nb:
        return na and nb
    if _is_bool(a) or _is_bool(b):
        return _is_bool(a) and _is_bool(b) and bool(a) == bool(b)
    if _is_number(a) and _is_number(b):
        fa, fb = float(a), float(b)
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        return abs(fa - fb) <= atol
    if _is_time(a) and _is_time(b):
        try:
            return pd.Timestamp(a) == pd.Timestamp(b)
        except (TypeError, ValueError):
            return False
    if isinstance(a, str) != isinstance(b, str):
        return False
    try:
        return bool(a == b)
    except Exception:
        return False


def equal_mask(a_values, b_values, atol: float = DEFAULT_ATOL) -> np.ndarray:
    """Element-wise value_equal for two equally long sequences."""
    return np.fromiter((value_equal(x, y, atol) for x, y in zip(a_values, b_values)), dtype=bool,
                       count=len(a_values))


def as_objects(s: pd.Series) -> list:
    """Cell values as plain Python/pandas scalars (categories -> their values)."""
    return s.astype(object).tolist()


def dtype_family(s: pd.Series) -> str:
    """§5.7 families: int, float, bool, string, category, datetime; anything else -> 'object' / dtype name."""
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
        return "datetime" if getattr(dt, "tz", None) is None else "datetime_tz"
    if pd.api.types.is_object_dtype(dt):
        return "string" if all(isinstance(v, str) for v in s.dropna()) else "object"
    if pd.api.types.is_string_dtype(dt):
        return "string"
    return str(dt)


def hashable_key(v):
    """Normalised, hashable form of a cell (for duplicate keys and content matching). Nulls are equal."""
    if is_null(v):
        return ("N",)
    if _is_bool(v):
        return ("B", bool(v))
    if _is_number(v):
        f = float(v)
        return ("X", round(f, 9) if math.isfinite(f) else f)
    if _is_time(v):
        try:
            return ("T", pd.Timestamp(v).isoformat())
        except (TypeError, ValueError):
            pass
    if isinstance(v, str):
        return ("S", v)
    try:
        hash(v)
        return ("O", v)
    except TypeError:
        return ("R", repr(v))


def row_keys(df: pd.DataFrame, columns) -> list[tuple]:
    if not len(columns):
        return [()] * len(df)
    cols = [[hashable_key(v) for v in as_objects(df[c])] for c in columns]
    return list(zip(*cols))
