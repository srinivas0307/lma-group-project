"""Canonical predicate keys for comparing a predicted contract with the gold one (guide §5.8).

1. Expand "*"/except and list targets into one predicate per column (partial credit).
2. Canonicalise conditions (§5.3): drop single-child all/any, remove double negation, sort children and
   `in` lists, numbers as numbers.
3. Key = (section, type, target column, canonical condition/where, op, value, keep, dtype, ordered,
   categories, fn, args). Missing optional fields = defaults. Floats rounded to 9 decimals. Ids ignored.
"""
import json

from .schema import Contract, expand_columns, load_contract


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    f = float(v)
    return int(f) if f.is_integer() else round(f, 9)


def _canon_value(v):
    if isinstance(v, list):
        return sorted((_canon_value(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(v, dict):
        return {k: _canon_value(x) for k, x in sorted(v.items())}
    return _num(v)


def canon_condition(c):
    """Condition model (or None) -> canonical plain structure."""
    if c is None:
        return None
    if c.all is not None or c.any is not None:
        op = "all" if c.all is not None else "any"
        kids = [canon_condition(k) for k in (c.all if c.all is not None else c.any)]
        flat = []
        for k in kids:                                   # all(all(a,b),c) == all(a,b,c)
            flat += k[op] if isinstance(k, dict) and op in k else [k]
        if len(flat) == 1:
            return flat[0]
        return {op: sorted(flat, key=lambda x: json.dumps(x, sort_keys=True, default=str))}
    if c.not_ is not None:
        inner = canon_condition(c.not_)
        if isinstance(inner, dict) and "not" in inner:
            return inner["not"]
        return {"not": inner}
    leaf = {"column": c.column, "op": c.op}
    if c.op not in ("is_null", "not_null"):
        leaf["value"] = _canon_value(c.value) if c.op in ("in", "not_in") else _num(c.value)
    return leaf


def _args(p):
    out = []
    for a in p.args:
        out.append({"col": a.col} if hasattr(a, "col") else {"lit": _canon_value(a.lit)})
    return out


def predicate_keys(section, p, before_columns):
    """One or more hashable keys for a predicate."""
    base = dict(section=section, type=p.type)
    if p.type in ("row_count", "null_count", "duplicate_count"):
        base.update(op=p.op, value=_num(p.value))
    if p.type == "duplicate_count":
        base["keep"] = p.keep
    if p.type in ("rows_removed_where", "rows_retained_where"):
        base["condition"] = canon_condition(p.condition)
    if p.type in ("column_values_unchanged", "column_value_relation"):
        base["where"] = canon_condition(p.where)
    if p.type == "column_dtype":
        base.update(dtype=p.dtype, ordered=p.ordered,
                    categories=list(p.categories) if p.categories is not None and p.ordered
                    else sorted(map(str, p.categories)) if p.categories is not None else None)
    if p.type == "column_value_relation":
        base.update(fn=p.fn, args=_args(p))
    if p.type in ("duplicate_count", "column_values_unchanged", "columns_unchanged"):
        targets = expand_columns(p.target, p.except_, before_columns)
    elif p.type in ("columns_added", "columns_removed"):
        targets = list(p.target)
    elif p.target is None:
        targets = [None]
    else:
        targets = [p.target]
    return [json.dumps(dict(base, target=t), sort_keys=True, default=str) for t in targets]


def contract_keys(contract, before_columns) -> dict:
    """{"must_change": set(keys), "must_preserve": set(keys)}"""
    c = contract if isinstance(contract, Contract) else load_contract(contract)
    out = {"must_change": set(), "must_preserve": set()}
    for section, _, p in c.predicates():
        out[section].update(predicate_keys(section, p, before_columns))
    return out
