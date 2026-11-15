# ===== Intent parser: shared input format + contract feedback checks =====

def describe_input(df):
    """Exactly the input format both 1.5B adapters were trained on (schema, missing counts, first 3 rows)."""
    lines = [f"Dataset: {len(df)} rows, {df.shape[1]} columns.", "", "Columns (name: dtype, missing values):"]
    lines += [f"- {c}: {df[c].dtype}, {int(df[c].isna().sum())} missing" for c in df.columns]
    return "\n".join(lines) + "\n\nFirst 3 rows (CSV):\n" + df.head(3).to_csv(index=False)


def contract_issues(contract, df):
    """Checks of a schema-valid contract against the INPUT data only (no reference, no gold).
    hard = the contract cannot be used as is; soft = suspicious, worth one round of feedback."""
    hard, soft = [], []
    cols = list(df.columns)
    need, new = contract_columns(contract)
    for sec in ("must_change", "must_preserve"):           # unknown input columns, reported per predicate
        for i, p in enumerate(contract[sec]):
            p_need, _ = contract_columns({"must_change": [p], "must_preserve": []})
            for c in sorted(p_need - set(cols)):
                if c in new and p["type"] in ("null_count", "column_dtype"):
                    continue   # checks on a column that another predicate of this contract creates
                if c in new:
                    hard.append(f"{sec}[{i}] {p['type']} refers to {c!r}, which is a NEW column, not an input column; conditions, "
                                f"keys and preserve predicates can only refer to input columns, so remove or fix this predicate")
                else:
                    close = difflib.get_close_matches(c, cols, n=3, cutoff=0.5)
                    hard.append(f"{sec}[{i}] {p['type']} refers to column {c!r}, which does not exist in the input data"
                                + (f" (did you mean {close}?)" if close else f"; input columns are {cols}"))
    added = {c for sec in ("must_change", "must_preserve") for p in contract[sec] if p["type"] == "columns_added" for c in p["target"]}
    for c in sorted(new - set(cols) - added):
        soft.append(f"target column {c!r} is not an input column; if a new column is intended, also state it with columns_added, "
                    f"otherwise use the exact input column name")
    if not contract["must_change"]:
        hard.append("must_change is empty; it must state what the request changes")
    # string literals in conditions must be values that occur in the column (exact spelling/case)
    def leaves(cond):
        if not isinstance(cond, dict): return []
        for k in ("all", "any"):
            if k in cond: return [x for c in cond[k] for x in leaves(c)]
        if "not" in cond: return leaves(cond["not"])
        return [cond]
    for sec in ("must_change", "must_preserve"):
        for p in contract[sec]:
            for k in ("condition", "where"):
                for lf in leaves(p.get(k)):
                    col = lf.get("column")
                    if col not in df.columns or df[col].dtype != object:
                        continue
                    vals = lf.get("value")
                    vals = vals if isinstance(vals, list) else ([vals] if isinstance(vals, str) else [])
                    present = set(df[col].dropna().unique().tolist())
                    for v in vals:
                        if isinstance(v, str) and v not in present:
                            elsewhere = [c2 for c2 in cols if c2 != col and df[c2].dtype == object and (df[c2] == v).any()]
                            if elsewhere:   # strong evidence of the wrong column
                                hard.append(f"value {v!r} never occurs in column {col!r}, but it occurs in column {elsewhere}; "
                                            f"use the column that contains this value")
                                continue
                            sample = sorted(map(str, present))[:15]
                            close = difflib.get_close_matches(v, list(map(str, present)), n=3, cutoff=0.6)
                            soft.append(f"value {v!r} never occurs in column {col!r}" +
                                        (f" (close matches: {close})" if close else f" (values include: {sample})"))
    soft = list(dict.fromkeys(soft))
    if hard:
        return list(dict.fromkeys(hard)), soft
    # internal consistency checks against the input data
    filled = set()
    for p in contract["must_change"]:
        if p["type"] == "column_value_relation" and p["fn"] != "null" and p.get("where"):
            if any(lf.get("column") == p["target"] and lf.get("op") == "is_null" for lf in leaves(p["where"])):
                filled.add(p["target"])
    for sec in ("must_change", "must_preserve"):
        for i, p in enumerate(contract[sec]):
            t = p["type"]
            if t == "rows_removed_where" and len(df) and eval_condition(df, p["condition"]).all():
                hard.append(f"{sec}[{i}] rows_removed_where: the condition matches every input row (all {len(df)} rows would be "
                            f"removed); check the columns and values of the condition")
            if t == "duplicate_count" and sec == "must_change":
                keys = expand_targets(p, cols)
                if all(k in cols for k in keys) and not df.duplicated(subset=keys).any():
                    hard.append(f"{sec}[{i}] duplicate_count: the input has no duplicate rows on key {keys}, so this deduplication "
                                f"removes nothing; use exactly the key columns the request names")
            if t == "null_count" and p["target"] in filled and p["op"] in ("unchanged", "increased"):
                hard.append(f"{sec}[{i}] null_count says {p['op']!r} for {p['target']!r}, but the contract fills the missing values "
                            f"of {p['target']!r}, so its null count decreases; fix or remove this predicate")
            if t == "column_dtype" and "categories" in p:
                keys_ = [json.dumps(x, default=str) for x in p["categories"]]
                dups = sorted({k for k in keys_ if keys_.count(k) > 1})
                if dups:
                    hard.append(f"{sec}[{i}] column_dtype: categories contain duplicates {dups}; each category must appear once")
                elif p["target"] in cols:
                    have = {str(_fmt(v)) for v in df[p["target"]].dropna().unique()}
                    cats = {str(x) for x in p["categories"]}
                    if have - cats:
                        soft.append(f"{sec}[{i}] column_dtype: categories omit values present in {p['target']!r}: {sorted(have - cats)[:15]}")
                    if cats - have:
                        soft.append(f"{sec}[{i}] column_dtype: categories include values that do not occur in {p['target']!r}: {sorted(cats - have)[:15]}")
    if hard:
        return list(dict.fromkeys(hard)), soft
    # the contract must not be satisfied by the unchanged input, and must be checkable on this data
    rep = verify(df, df.copy(), contract)
    if rep["verdict"] == "PASS":
        hard.append("this contract is already satisfied by the unchanged input data, so must_change does not capture "
                    "the requested change (every must_change predicate must be something the correct transformation actually changes)")
    for r in rep["predicates"]:
        if r["status"] == UNCERTAIN:
            hard.append(f"{r['section']}[{r['index']}] {r['type']}: {r['detail']}")
    return list(dict.fromkeys(hard)), soft


def better_attempt(best, contract, errors, hard, soft):
    """Keep the best parser attempt: clean < warnings only < hard issues < invalid (fewer problems break ties;
    an earlier attempt wins a full tie). A worse later retry (e.g. a sampled reply with broken JSON) never
    replaces a usable earlier contract."""
    rank = (3 if errors or contract is None else 2 if hard else 1 if soft else 0, len(errors or (hard + soft)))
    if best is None or rank < best["rank"]:
        return {"rank": rank, "contract": contract, "errors": errors, "hard": hard, "soft": soft}
    return best


def describe_predicate(p):
    """Plain-English rendering of one predicate (used by the planner/generator and the repair agent)."""
    def cond(c):
        if c is None: return "all rows"
        if "all" in c: return "(" + " AND ".join(cond(x) for x in c["all"]) + ")"
        if "any" in c: return "(" + " OR ".join(cond(x) for x in c["any"]) + ")"
        if "not" in c: return f"NOT {cond(c['not'])}"
        if c["op"] in NULL_OPS: return f"{c['column']} {'is missing' if c['op'] == 'is_null' else 'is not missing'}"
        return f"{c['column']} {c['op']} {json.dumps(c.get('value'))}"
    def arg(a): return a["col"] if "col" in a else json.dumps(a["lit"])
    t = p["type"]
    if t == "row_count": return f"row count {p['op']}" + (f" {p['value']}" if "value" in p else "") + " (relative to the input)"
    if t == "rows_removed_where": return f"remove every row where {cond(p['condition'])}"
    if t == "rows_retained_where": return f"keep every row where {cond(p['condition'])}"
    if t == "duplicate_count":
        keys = p["target"] if p["target"] != "*" else f"all columns except {p.get('except', [])}"
        return f"no duplicate rows on key {keys}" + (f"; keep the {p['keep']} row of each key group (original order)" if "keep" in p else "")
    if t == "null_count": return f"missing values in {p['target']}: {p['op']}" + (f" {p['value']}" if "value" in p else "")
    if t == "column_values_unchanged":
        return f"values of {p['target'] if p['target'] != '*' else 'all columns except ' + str(p.get('except', []))} unchanged" + (f" where {cond(p['where'])}" if p.get("where") else "")
    if t == "columns_unchanged":
        return f"columns {p['target'] if p['target'] != '*' else 'all except ' + str(p.get('except', []))} keep the same dtype and values"
    if t == "column_dtype":
        return f"{p['target']} must have dtype {p['dtype']}" + (f", ordered={p['ordered']}" if "ordered" in p else "") + (f", categories={p['categories']}" if "categories" in p else "")
    if t == "column_value_relation":
        return f"{p['target']} = {p['fn']}({', '.join(arg(a) for a in p.get('args', []))})" + (f" where {cond(p['where'])}" if p.get("where") else " on every row")
    if t == "columns_added": return f"add columns {p['target']}"
    if t == "columns_removed": return f"remove columns {p['target']}"
    return json.dumps(p)


def describe_contract(contract):
    lines = ["MUST CHANGE:"] + [f"  - {describe_predicate(p)}" for p in contract["must_change"]]
    lines += ["MUST PRESERVE:"] + [f"  - {describe_predicate(p)}" for p in contract["must_preserve"]]
    return "\n".join(lines)


def tokens_of(calls):
    return sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls)
