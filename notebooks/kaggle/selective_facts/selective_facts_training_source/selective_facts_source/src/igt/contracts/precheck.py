"""Contract pre-check against the input data (extension #9, version `pc1`; docs/extensions.md).

    issues = precheck(contract_dict, before_df)     # [] = no issue; else a list of dicts(check, where, message)

Deterministic checks of a contract that already passed vocabulary validation, run on the BEFORE data only (never on
a candidate's output, never on the answer key). Each issue names something the contract asks for that the data
can't support:

    zero_match       a must_change condition (rows_removed_where, or the `where` of a must_change relation) matches 0 rows
    stat_name        a value that is the name of a statistic ("median", "mean", ...) instead of a value
    type_mismatch    text compared with a numeric column, or a number compared with a text column
    absent_value     an `==` / `in` text value that doesn't occur in the column
    unknown_dup_key  a duplicate_count key column that isn't in the data

Off by default: nothing in the existing systems calls this module. It is used by eval/ext_precheck.py (pc1) and
eval/ext_reask.py --precheck (pc1+rq1).
"""
import numpy as np
import pandas as pd

from igt.contracts.schema import ColArg, LitArg, expand_columns, load_contract

PRECHECK_VERSION = "pc1"
CHECKS = ("zero_match", "stat_name", "type_mismatch", "absent_value", "unknown_dup_key")
STAT_NAMES = {"median", "mean", "mode", "average", "avg", "most common", "most_common", "most frequent",
              "most_frequent", "the median", "the mean", "the mode", "the average", "the most common value"}


def _is_stat(v):
    return isinstance(v, str) and " ".join(v.strip().lower().replace("_", " ").split()) in \
        {" ".join(s.replace("_", " ").split()) for s in STAT_NAMES}


def _numeric(s):
    return pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)


def _texty(s):
    return s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s.dtype)


def _is_num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _leaves(cond):
    if cond is None:
        return
    if cond.column is not None:
        yield cond
    for k in (cond.all or []) + (cond.any or []) + ([cond.not_] if cond.not_ else []):
        yield from _leaves(k)


def _leaf_issues(leaf, df, where):
    out = []
    if leaf.column not in df.columns:
        return out                                   # unknown columns are already UNCERTAIN in the verifier
    s = df[leaf.column]
    vals = leaf.value if isinstance(leaf.value, list) else [leaf.value] if leaf.value is not None else []
    for v in vals:
        if _is_stat(v):
            out.append(dict(check="stat_name", where=where,
                            message=f"condition on '{leaf.column}' compares with {v!r}, the name of a statistic, not a value"))
        elif isinstance(v, str) and _numeric(s):
            try:
                float(v)
                hint = " (a number written as text never equals a number)"
            except ValueError:
                hint = ""
            out.append(dict(check="type_mismatch", where=where,
                            message=f"text {v!r} compared with numeric column '{leaf.column}'{hint}"))
        elif _is_num(v) and _texty(s) and pd.to_numeric(s.dropna(), errors="coerce").notna().sum() == 0:
            out.append(dict(check="type_mismatch", where=where,
                            message=f"number {v!r} compared with text column '{leaf.column}'"))
        elif isinstance(v, str) and _texty(s) and leaf.op in ("==", "in") and \
                v not in set(s.dropna().astype(str).unique()):
            out.append(dict(check="absent_value", where=where,
                            message=f"value {v!r} does not occur in column '{leaf.column}'"))
    return out


def _matches(cond, df):
    from igt.verifier.conditions import ConditionError, evaluate
    try:
        return int(np.asarray(evaluate(cond, df)).sum())
    except (ConditionError, TypeError, ValueError):
        return None                                  # the verifier reports this itself


def precheck(contract, before):
    """contract: dict (valid under vocabulary v1.0) or Contract; before: the input DataFrame. -> list of issues."""
    c = load_contract(contract) if isinstance(contract, dict) else contract
    issues = []
    for direction, pid, p in c.predicates():
        where = f"{direction}.{pid} ({p.type})"
        cond = getattr(p, "condition", None) or getattr(p, "where", None)
        for leaf in _leaves(cond):
            issues += _leaf_issues(leaf, before, where)
        if direction == "must_change" and cond is not None and p.type in ("rows_removed_where", "column_value_relation"):
            n = _matches(cond, before)
            if n == 0:
                issues.append(dict(check="zero_match", where=where,
                                   message="the condition matches 0 rows of the input, so the requested change can't happen"))
        if p.type == "column_value_relation":
            for a in p.args:
                if isinstance(a, LitArg) and _is_stat(a.lit):
                    issues.append(dict(check="stat_name", where=where,
                                       message=f"literal {a.lit!r} is the name of a statistic; the contract needs the "
                                               f"actual value computed from the data"))
            if p.fn == "map" and len(p.args) == 2 and isinstance(p.args[0], ColArg) and isinstance(p.args[1].lit, dict):
                for v in p.args[1].lit.values():
                    if _is_stat(v):
                        issues.append(dict(check="stat_name", where=where, message=f"mapped value {v!r} is a statistic name"))
        if p.type == "duplicate_count" and p.target != "*":
            keys = [p.target] if isinstance(p.target, str) else list(p.target)
            missing = [k for k in keys if k not in before.columns]
            if missing:
                issues.append(dict(check="unknown_dup_key", where=where,
                                   message=f"duplicate key column(s) {missing} not in the data"))
        if p.type == "duplicate_count" and p.target == "*" and p.except_:
            if not expand_columns("*", p.except_, before.columns):
                issues.append(dict(check="unknown_dup_key", where=where, message="the duplicate key excludes every column"))
    seen, out = set(), []
    for i in issues:                                 # one issue per (check, where, message)
        k = (i["check"], i["where"], i["message"])
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out


def issues_text(issues):
    return "\n".join(f"- {i['where']}: {i['message']}" for i in issues)
