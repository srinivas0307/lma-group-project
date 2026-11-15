# ===== Evaluator (answer key: reference output + gold contract; never visible to the system) =====

def outputs_match(cand, ref, atol=1e-6):
    """Is the candidate output the same table as the reference output?
    Same row ids (index labels), same column set, same dtype family per column, equal values
    (null == null, numbers within atol, categories compared by value + ordered flag + category list)."""
    if not isinstance(cand, pd.DataFrame):
        return False, "candidate is not a DataFrame"
    if set(cand.columns) != set(ref.columns) or cand.columns.duplicated().any():
        return False, f"columns differ: extra={sorted(set(cand.columns) - set(ref.columns))} missing={sorted(set(ref.columns) - set(cand.columns))}"
    if cand.index.has_duplicates or set(cand.index) != set(ref.index):
        return False, f"row ids differ: candidate {len(cand)} rows, reference {len(ref)} rows"
    c = cand.loc[ref.index, list(ref.columns)]
    for col in ref.columns:
        fc, fr = dtype_family(c[col]), dtype_family(ref[col])
        if fc != fr:
            return False, f"column {col!r}: dtype family {fc} vs reference {fr}"
        if fr == "category":
            cc, rc = c[col].cat, ref[col].cat
            if bool(cc.ordered) != bool(rc.ordered):
                return False, f"column {col!r}: ordered flag differs"
            a, b = [_fmt(x) for x in cc.categories], [_fmt(x) for x in rc.categories]
            if (rc.ordered and a != b) or (not rc.ordered and sorted(map(str, a)) != sorted(map(str, b))):
                return False, f"column {col!r}: categories differ"
        eq = series_equal_mask(c[col], ref[col], atol)
        if not eq.all():
            r = eq.index[~eq.values][0]
            return False, f"column {col!r}: {int((~eq).sum())} values differ (row {r}: {_fmt(c.at[r, col])!r} vs {_fmt(ref.at[r, col])!r})"
    return True, "identical"


_REF_CACHE = {}

def reference_output(task):
    if task["id"] not in _REF_CACHE:
        ex = run_in_sandbox(task["reference_code"], load_before(task))
        assert ex["ok"], f"reference failed for {task['id']}: {ex['error']}"
        _REF_CACHE[task["id"]] = ex["output"]
    return _REF_CACHE[task["id"]]


# ---------------------------------------------------------------- contract matching (spec section 5.8)
def _canon_value(v):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 9)
    if isinstance(v, list):
        return sorted((_canon_value(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True, default=str))
    if isinstance(v, dict):
        return {k: _canon_value(x) for k, x in sorted(v.items())}
    return str(v)


def canon_condition(c):
    if c is None:
        return None
    if "all" in c or "any" in c:
        k = "all" if "all" in c else "any"
        kids = [canon_condition(x) for x in c[k]]
        flat = []
        for x in kids:
            flat += x[k] if isinstance(x, dict) and k in x else [x]
        if len(flat) == 1:
            return flat[0]
        return {k: sorted(flat, key=lambda x: json.dumps(x, sort_keys=True, default=str))}
    if "not" in c:
        inner = canon_condition(c["not"])
        if isinstance(inner, dict) and "not" in inner:
            return inner["not"]
        return {"not": inner}
    out = {"column": c["column"], "op": c["op"]}
    if "value" in c:
        out["value"] = _canon_value(c["value"])
    return out


def predicate_keys(contract, section, before_cols):
    keys = []
    for p in contract.get(section) or []:
        t = p["type"]
        base = {k: v for k, v in p.items() if k not in ("id", "atol", "target", "except", "condition", "where")}
        for k in ("condition", "where"):
            if p.get(k) is not None:
                base[k] = canon_condition(p[k])
        if t in ("columns_unchanged", "column_values_unchanged", "columns_added", "columns_removed"):
            targets = expand_targets(p, before_cols)
        elif t == "duplicate_count":
            targets = [tuple(sorted(expand_targets(p, before_cols)))]
        else:
            targets = [p.get("target")]
        for tg in targets:
            keys.append(json.dumps({"section": section, "target": tg, **_canon_value(base)}, sort_keys=True, default=str))
    return keys


def contract_scores(pred, gold, before_cols):
    out = {}
    for sec in ("must_change", "must_preserve"):
        g = predicate_keys(gold, sec, before_cols)
        p = predicate_keys(pred, sec, before_cols) if pred else []
        gs, ps = set(g), set(p)
        tp = len(gs & ps)
        prec = tp / len(ps) if ps else 0.0
        rec = tp / len(gs) if gs else 1.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[sec] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "exact": gs == ps}
    out["exact_match"] = out["must_change"]["exact"] and out["must_preserve"]["exact"]
    return out
