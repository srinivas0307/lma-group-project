"""Evaluate a condition tree on the BEFORE data (§5.3). A null cell makes every comparison false,
including `!=`, `in` and `not_in`; only `is_null` / `not_null` look at nulls."""
import numpy as np
import pandas as pd

from .values import is_null


class ConditionError(ValueError):
    pass


def _coerce(value, s: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(s.dtype) and isinstance(value, str):
        return pd.Timestamp(value)
    return value


def _leaf(df, c) -> np.ndarray:
    if c.column not in df.columns:
        raise ConditionError(f"condition column '{c.column}' not in data")
    s = df[c.column]
    notnull = ~s.isna().to_numpy()
    op = c.op
    if op == "is_null":
        return ~notnull
    if op == "not_null":
        return notnull
    if op in ("in", "not_in"):
        vals = [_coerce(v, s) for v in c.value]
        member = s.astype(object).map(lambda x: any(_eq(x, v) for v in vals)).to_numpy(dtype=bool)
        return (member if op == "in" else ~member) & notnull
    v = _coerce(c.value, s)
    out = np.zeros(len(s), dtype=bool)
    for i, x in enumerate(s.astype(object).tolist()):
        if is_null(x):
            continue
        try:
            out[i] = bool({"==": _eq(x, v), "!=": not _eq(x, v), "<": x < v, "<=": x <= v,
                           ">": x > v, ">=": x >= v}[op])
        except TypeError:
            raise ConditionError(f"can't compare column '{c.column}' value {x!r} with {v!r} using {op}") from None
    return out


def _eq(x, v) -> bool:
    if isinstance(x, str) != isinstance(v, str):
        return False
    try:
        return bool(x == v)
    except Exception:
        return False


def evaluate(cond, df: pd.DataFrame) -> np.ndarray:
    """Boolean mask over df rows (positional)."""
    if cond.all is not None:
        m = np.ones(len(df), dtype=bool)
        for k in cond.all:
            m &= evaluate(k, df)
        return m
    if cond.any is not None:
        m = np.zeros(len(df), dtype=bool)
        for k in cond.any:
            m |= evaluate(k, df)
        return m
    if cond.not_ is not None:
        return ~evaluate(cond.not_, df)
    return _leaf(df, cond)
