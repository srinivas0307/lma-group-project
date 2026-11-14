"""Relation functions for `column_value_relation` (§5.6). Each computes the expected after-value of the
target column from the BEFORE row. Null input -> null output (except const / map of unlisted values)."""
import numpy as np
import pandas as pd

from igt.contracts.schema import ColArg

from .values import is_null


def _arg(a, before: pd.DataFrame):
    """Column argument -> list of before values; literal -> scalar."""
    return before[a.col].astype(object).tolist() if isinstance(a, ColArg) else a.lit


def _vec(x, n):
    return x if isinstance(x, list) else [x] * n


def _arith(op, a, b):
    if is_null(a) or is_null(b):
        return None
    try:
        if op == "div":
            return None if b == 0 else a / b
        return {"add": a + b, "sub": a - b, "mul": a * b}[op]
    except TypeError:
        return None


def _cast(v, dtype, fmt=None):
    if is_null(v):
        return None
    try:
        if dtype == "datetime":
            return pd.to_datetime(v, format=fmt or "ISO8601")
        if dtype in ("int", "float"):
            x = pd.to_numeric(v)
            if dtype == "int":
                return int(x) if float(x).is_integer() else None
            return float(x)
        if dtype == "string":
            return v if isinstance(v, str) else str(v)
        if dtype == "bool":
            if isinstance(v, str):
                t = v.strip().lower()
                return True if t in ("true", "1", "yes") else False if t in ("false", "0", "no") else None
            return bool(v)
        if dtype == "category":
            return v
    except (ValueError, TypeError, OverflowError):
        return None
    return None


def _str_fn(fn, v):
    if is_null(v) or not isinstance(v, str):
        return None if is_null(v) else v
    return {"lower": v.lower(), "upper": v.upper(), "strip": v.strip()}[fn]


def expected_values(pred, before: pd.DataFrame) -> list:
    """Expected after-values of pred.target, one per before row (positional)."""
    n = len(before)
    fn, args = pred.fn, [_arg(a, before) for a in pred.args]
    if fn == "identity":
        return list(args[0])
    if fn == "const":
        return [args[0]] * n
    if fn == "null":
        return [None] * n
    if fn in ("add", "sub", "mul", "div"):
        a, b = _vec(args[0], n), _vec(args[1], n)
        return [_arith(fn, x, y) for x, y in zip(a, b)]
    if fn == "round":
        nd = int(args[1])
        return [None if is_null(v) else float(np.round(float(v), nd)) for v in args[0]]
    if fn == "clip":
        lo, hi = args[1], args[2]
        out = []
        for v in args[0]:
            if is_null(v):
                out.append(None)
                continue
            if lo is not None and v < lo:
                v = lo
            if hi is not None and v > hi:
                v = hi
            out.append(v)
        return out
    if fn in ("lower", "upper", "strip"):
        return [_str_fn(fn, v) for v in args[0]]
    if fn == "map":
        mapping = args[1]
        return [None if is_null(v) else mapping.get(v, mapping.get(str(v), v)) for v in args[0]]
    if fn == "cast":
        fmt = args[2] if len(args) > 2 else None
        return [_cast(v, args[1], fmt) for v in args[0]]
    raise ValueError(f"unknown function {fn}")
