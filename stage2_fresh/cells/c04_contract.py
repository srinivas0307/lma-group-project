# ===== Intent contract: JSON extraction, schema validation, condition semantics, value equality =====

DTYPES = ("int", "float", "bool", "string", "category", "datetime")
CMP_OPS = ("==", "!=", "<", "<=", ">", ">=")
SET_OPS = ("in", "not_in")
NULL_OPS = ("is_null", "not_null")
COUNT_REL = ("unchanged", "decreased", "increased")
COUNT_ABS = ("eq", "lt", "le", "gt", "ge")
# fn -> argument kinds; "any" = col or lit
FN_SIGNATURES = {
    "identity": [["col"]], "const": [["lit"]], "null": [[]],
    "add": [["any", "any"]], "sub": [["any", "any"]], "mul": [["any", "any"]], "div": [["any", "any"]],
    "round": [["col", "lit"]], "clip": [["col", "lit", "lit"]],
    "lower": [["col"]], "upper": [["col"]], "strip": [["col"]],
    "map": [["col", "lit"]], "cast": [["col", "lit"], ["col", "lit", "lit"]],
}
PRED_FIELDS = {
    "row_count": {"op", "value"},
    "rows_removed_where": {"condition"},
    "rows_retained_where": {"condition"},
    "duplicate_count": {"except", "op", "value", "keep"},
    "null_count": {"op", "value"},
    "column_values_unchanged": {"except", "where"},
    "columns_unchanged": {"except"},
    "column_dtype": {"dtype", "ordered", "categories"},
    "column_value_relation": {"fn", "args", "where", "atol"},
    "columns_added": set(),
    "columns_removed": set(),
}
COMMON_FIELDS = {"type", "target", "id"}


class ContractParseError(Exception):
    pass


def extract_json_objects(text):
    """All top-level JSON objects in a model reply: fenced blocks first, then bare balanced braces."""
    found = []
    for block in re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.S):
        try:
            found.append(json.loads(block.strip()))
        except Exception:
            pass
    if found:
        return found
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    found.append(json.loads(text[start:i + 1]))
                except Exception:
                    pass
    return found


def json_block_errors(text):
    """Parse errors of the ```json blocks in a reply (fed back to the parser instead of 'no JSON found')."""
    errs = []
    for block in re.findall(r"```(?:json|JSON)?\s*(.*?)(?:```|$)", text, flags=re.S):
        try:
            json.loads(block.strip())
        except json.JSONDecodeError as e:
            lines = block.strip().splitlines()
            bad = lines[e.lineno - 1].strip()[:160] if 0 < e.lineno <= len(lines) else ""
            errs.append(f"the json block is not valid JSON: {e.msg} at line {e.lineno} column {e.colno} (near: {bad!r}); "
                        f"write plain JSON only (no '&', comments or Python expressions)")
    return errs


def no_contract_error(text, truncated):
    if truncated:
        return "the reply was cut off before the JSON was complete; write the contract more compactly"
    errs = json_block_errors(text)
    return errs[0] if errs else "no JSON contract found in the reply"


# ---------------------------------------------------------------- schema validation
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_scalar(v):
    return v is None or isinstance(v, (str, int, float, bool))


def validate_condition(cond, path, errors):
    if not isinstance(cond, dict):
        errors.append(f"{path}: condition must be an object"); return
    for bad in ("and", "or"):
        if bad in cond:
            errors.append(f"{path}: there is no '{bad}' key; combine conditions with \"all\"/\"any\"/\"not\"")
            return
    keys = set(cond)
    if keys & {"all", "any"}:
        k = "all" if "all" in cond else "any"
        if keys != {k} or not isinstance(cond[k], list) or not cond[k]:
            errors.append(f"{path}.{k}: must be the only key and a non-empty list of conditions"); return
        for i, c in enumerate(cond[k]):
            validate_condition(c, f"{path}.{k}[{i}]", errors)
        return
    if "not" in keys:
        if keys != {"not"}:
            errors.append(f"{path}.not: must be the only key"); return
        validate_condition(cond["not"], f"{path}.not", errors); return
    if not isinstance(cond.get("column"), str):
        errors.append(f"{path}: leaf condition needs a string \"column\""); return
    op = cond.get("op")
    extra = keys - {"column", "op", "value"}
    if extra:
        errors.append(f"{path}: unknown keys {sorted(extra)} in condition")
    if op in CMP_OPS:
        if "value" not in cond or not _is_scalar(cond["value"]) or cond["value"] is None:
            errors.append(f"{path}: op {op!r} needs a non-null scalar \"value\"")
    elif op in SET_OPS:
        if not isinstance(cond.get("value"), list) or not all(_is_scalar(v) for v in cond["value"]):
            errors.append(f"{path}: op {op!r} needs a list \"value\"")
    elif op in NULL_OPS:
        if "value" in cond:
            errors.append(f"{path}: op {op!r} takes no \"value\"")
    else:
        errors.append(f"{path}: unknown op {op!r}; use one of {list(CMP_OPS + SET_OPS + NULL_OPS)}")


def _validate_count(p, path, errors):
    op = p.get("op")
    if op in COUNT_REL and p.get("value", 0) is None:
        p.pop("value")                     # harmless "value": null on a relative op
    if op in COUNT_REL:
        if "value" in p:
            errors.append(f"{path}: relative op {op!r} takes no \"value\"")
    elif op in COUNT_ABS:
        if not _is_int(p.get("value")):
            errors.append(f"{path}: op {op!r} needs an integer \"value\"")
    else:
        errors.append(f"{path}: op must be one of {list(COUNT_REL + COUNT_ABS)}, got {op!r}")


def _validate_cols(target, path, errors, allow_star=True, allow_str=True):
    if target == "*" and allow_star:
        return
    if isinstance(target, str) and allow_str and target != "*":
        return
    if isinstance(target, list) and target and all(isinstance(c, str) for c in target):
        return
    star = '"*", ' if allow_star else ''
    errors.append(f"{path}.target: expected {star}a column name or a list of column names, got {target!r}")


def validate_predicate(p, path, errors):
    if not isinstance(p, dict):
        errors.append(f"{path}: predicate must be an object"); return
    t = p.get("type")
    if t not in PRED_FIELDS:
        errors.append(f"{path}: unknown predicate type {t!r}; use one of {sorted(PRED_FIELDS)}"); return
    path = f"{path}.{t}"
    extra = set(p) - PRED_FIELDS[t] - COMMON_FIELDS
    if extra:
        errors.append(f"{path}: unknown fields {sorted(extra)}")
    tgt = p.get("target")
    if t in ("row_count", "rows_removed_where", "rows_retained_where"):
        if tgt is not None:
            errors.append(f"{path}.target: must be null")
    if t in ("row_count", "null_count", "duplicate_count"):
        _validate_count(p, path, errors)
    if t in ("rows_removed_where", "rows_retained_where"):
        validate_condition(p.get("condition"), f"{path}.condition", errors)
    if t == "duplicate_count":
        _validate_cols(tgt, path, errors, allow_star=True, allow_str=False)
        if "keep" in p:
            if p["keep"] not in ("first", "last"):
                errors.append(f"{path}.keep: must be \"first\" or \"last\"")
            if not (p.get("op") == "eq" and p.get("value") == 0):
                errors.append(f"{path}: \"keep\" appears only with \"op\": \"eq\", \"value\": 0")
    if t == "null_count" and not isinstance(tgt, str):
        errors.append(f"{path}.target: must be one column name")
    if t in ("column_values_unchanged", "columns_unchanged"):
        _validate_cols(tgt, path, errors)
    if "except" in p and not (isinstance(p["except"], list) and all(isinstance(c, str) for c in p["except"])):
        errors.append(f"{path}.except: must be a list of column names")
    if "where" in p and p["where"] is not None:
        validate_condition(p["where"], f"{path}.where", errors)
    if t == "column_dtype":
        if not isinstance(tgt, str):
            errors.append(f"{path}.target: must be one column name")
        if p.get("dtype") not in DTYPES:
            errors.append(f"{path}.dtype: must be one of {list(DTYPES)} (never a pandas name such as int64/object), got {p.get('dtype')!r}")
        if p.get("dtype") != "category" and ("ordered" in p or "categories" in p):
            errors.append(f"{path}: \"ordered\"/\"categories\" appear only with \"dtype\": \"category\"")
        if "ordered" in p and not isinstance(p["ordered"], bool):
            errors.append(f"{path}.ordered: must be true or false")
        if "categories" in p and not isinstance(p["categories"], list):
            errors.append(f"{path}.categories: must be a list")
    if t == "column_value_relation":
        if not isinstance(tgt, str):
            errors.append(f"{path}.target: must be one column name")
        fn = p.get("fn")
        args = p.get("args", [])
        if fn not in FN_SIGNATURES:
            errors.append(f"{path}.fn: unknown function {fn!r}; use one of {sorted(FN_SIGNATURES)} (no mean/median/mode/replace: write the value as a lit)")
            return
        if not isinstance(args, list):
            errors.append(f"{path}.args: must be a list"); return
        kinds = []
        for i, a in enumerate(args):
            if not (isinstance(a, dict) and len(a) == 1 and next(iter(a)) in ("col", "lit")):
                errors.append(f"{path}.args[{i}]: every arg is its own object with exactly one key, {{\"col\": name}} or {{\"lit\": value}}; got {a!r}")
                return
            kinds.append(next(iter(a)))
            if kinds[-1] == "col" and not isinstance(a["col"], str):
                errors.append(f"{path}.args[{i}].col: must be a column name")
        if not any(len(sig) == len(kinds) and all(s in ("any", k) for s, k in zip(sig, kinds)) for sig in FN_SIGNATURES[fn]):
            errors.append(f"{path}: fn {fn} expects args {' or '.join(str(s) for s in FN_SIGNATURES[fn])}, got {kinds}")
            return
        if fn == "clip" and not all(a["lit"] is None or (isinstance(a["lit"], (int, float)) and not isinstance(a["lit"], bool)) for a in args[1:]):
            errors.append(f"{path}: clip bounds must be numbers or null")
        if fn == "round" and not _is_int(args[1]["lit"]):
            errors.append(f"{path}: round needs an integer number of decimals")
        if fn == "map" and not isinstance(args[1]["lit"], dict):
            errors.append(f"{path}: map needs {{\"lit\": {{\"from\": \"to\", ...}}}}")
        if fn == "cast":
            if args[1]["lit"] not in DTYPES:
                errors.append(f"{path}: fn cast: dtype must be one of {list(DTYPES)}")
            if len(args) == 3 and not isinstance(args[2]["lit"], str):
                errors.append(f"{path}: cast format must be a string like \"%d/%m/%Y\"")
        if "atol" in p and not isinstance(p["atol"], (int, float)):
            errors.append(f"{path}.atol: must be a number")
    if t in ("columns_added", "columns_removed"):
        _validate_cols(tgt, path, errors, allow_star=False)
        if isinstance(tgt, list) and "*" in tgt:
            errors.append(f"{path}.target: list the column names explicitly (\"*\" is not allowed here)")


def normalize_contract(obj):
    """Validate a parsed JSON object as an intent contract. Returns (contract, errors)."""
    errors = []
    if not isinstance(obj, dict):
        return None, ["the reply is not a JSON object"]
    if "action" in obj or "tool" in obj:
        return None, ["expected a contract, got a tool call or action"]
    extra = set(obj) - {"contract_version", "must_change", "must_preserve"}
    if extra:
        errors.append(f"unknown top-level keys {sorted(extra)}")
    c = {"contract_version": str(obj.get("contract_version", "1.0")),
         "must_change": obj.get("must_change"), "must_preserve": obj.get("must_preserve")}
    for sec in ("must_change", "must_preserve"):
        if not isinstance(c[sec], list):
            errors.append(f"{sec}: must be a list"); continue
        c[sec] = [dict(p) if isinstance(p, dict) else p for p in c[sec]]
        for i, p in enumerate(c[sec]):
            if isinstance(p, dict):
                if p.get("type") in ("row_count", "rows_removed_where", "rows_retained_where"):
                    p.setdefault("target", None)
                if p.get("type") in ("columns_added", "columns_removed") and isinstance(p.get("target"), str):
                    p["target"] = [p["target"]]
            validate_predicate(p, f"{sec}[{i}]", errors)
    return (c if not errors else None), errors


# ---------------------------------------------------------------- values, dtypes, conditions
def is_null(v):
    if v is None or v is pd.NaT or v is pd.NA:
        return True
    try:
        return bool(pd.isna(v)) if not isinstance(v, (list, dict, tuple, set, np.ndarray)) else False
    except (TypeError, ValueError):
        return False


def _is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _is_datetime(v):
    return isinstance(v, (pd.Timestamp, np.datetime64)) or type(v).__name__ == "datetime"


def scalar_equal(a, b, atol=1e-9):
    na, nb = is_null(a), is_null(b)
    if na or nb:
        return na and nb
    if _is_number(a) and _is_number(b):
        fa, fb = float(a), float(b)
        if math.isinf(fa) or math.isinf(fb):
            return fa == fb
        return abs(fa - fb) <= atol
    if _is_datetime(a) and _is_datetime(b):
        return pd.Timestamp(a) == pd.Timestamp(b)
    try:
        r = a == b
        return bool(r) if isinstance(r, (bool, np.bool_)) else False
    except Exception:
        return False


def series_equal_mask(x, y, atol=1e-9):
    """Element-wise value equality of two aligned Series (null == null, numbers by value within atol)."""
    xs = x.astype(object) if isinstance(x.dtype, pd.CategoricalDtype) else x
    ys = y.astype(object) if isinstance(y.dtype, pd.CategoricalDtype) else y
    if pd.api.types.is_numeric_dtype(xs) and pd.api.types.is_numeric_dtype(ys) \
            and not pd.api.types.is_bool_dtype(xs) and not pd.api.types.is_bool_dtype(ys):
        xf = pd.to_numeric(xs, errors="coerce").astype(float)
        yf = pd.to_numeric(ys, errors="coerce").astype(float)
        both_null = xf.isna() & yf.isna()
        with np.errstate(invalid="ignore"):
            close = (xf - yf).abs() <= atol
        close |= (xf == yf)  # infinities
        return (both_null | close.fillna(False)).astype(bool)
    return pd.Series([scalar_equal(a, b, atol) for a, b in zip(xs.tolist(), ys.tolist())], index=x.index, dtype=bool)


def dtype_family(s):
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
    if pd.api.types.is_string_dtype(dt) and not pd.api.types.is_object_dtype(dt):
        return "string"
    if pd.api.types.is_object_dtype(dt):
        vals = s.dropna()
        if all(isinstance(v, str) for v in vals.tolist()):
            return "string"
        return "object"
    return str(dt)


class MissingColumn(Exception):
    def __init__(self, column):
        super().__init__(column)
        self.column = column


def condition_columns(cond, out=None):
    out = set() if out is None else out
    if not isinstance(cond, dict):
        return out
    for k in ("all", "any"):
        if k in cond:
            for c in cond[k]:
                condition_columns(c, out)
            return out
    if "not" in cond:
        return condition_columns(cond["not"], out)
    if "column" in cond:
        out.add(cond["column"])
    return out


def _coerce_value_for(s, v):
    if pd.api.types.is_datetime64_any_dtype(s.dtype) and isinstance(v, str):
        try:
            return pd.Timestamp(v)
        except Exception:
            return v
    return v


def eval_condition(df, cond):
    """Boolean Series over df rows. A null cell makes every comparison false (also != and not_in)."""
    if "all" in cond:
        m = pd.Series(True, index=df.index)
        for c in cond["all"]:
            m &= eval_condition(df, c)
        return m
    if "any" in cond:
        m = pd.Series(False, index=df.index)
        for c in cond["any"]:
            m |= eval_condition(df, c)
        return m
    if "not" in cond:
        return ~eval_condition(df, cond["not"])
    col, op = cond["column"], cond["op"]
    if col not in df.columns:
        raise MissingColumn(col)
    s = df[col]
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    notnull = s.notna()
    if op == "is_null":
        return ~notnull
    if op == "not_null":
        return notnull
    if op in SET_OPS:
        vals = cond["value"]
        hit = pd.Series([any(scalar_equal(x, v, 0) for v in vals) if not is_null(x) else False for x in s.tolist()], index=s.index)
        return (hit if op == "in" else ~hit) & notnull
    v = _coerce_value_for(s, cond["value"])
    def cmp(x):
        if is_null(x):
            return False
        try:
            if op == "==": return scalar_equal(x, v, 0)
            if op == "!=": return not scalar_equal(x, v, 0)
            if op == "<": return x < v
            if op == "<=": return x <= v
            if op == ">": return x > v
            if op == ">=": return x >= v
        except Exception:
            return False
    if pd.api.types.is_numeric_dtype(s) and _is_number(v):
        f = {"==": s.eq, "!=": s.ne, "<": s.lt, "<=": s.le, ">": s.gt, ">=": s.ge}[op]
        return f(v).fillna(False).astype(bool) & notnull
    return pd.Series([bool(cmp(x)) for x in s.tolist()], index=s.index, dtype=bool)


def expand_targets(p, before_cols):
    """Expand a target ("*" + except, a name, or a list) to a list of column names."""
    tgt = p.get("target")
    if tgt == "*":
        exc = set(p.get("except") or [])
        return [c for c in before_cols if c not in exc]
    if isinstance(tgt, str):
        return [tgt]
    return list(tgt or [])


def contract_columns(contract):
    """Columns a contract refers to, split into 'must exist in the input' and 'may be new'."""
    need, new, dtype_targets = set(), set(), set()
    for sec in ("must_change", "must_preserve"):
        for p in contract[sec]:
            t = p["type"]
            for k in ("condition", "where"):
                if p.get(k):
                    need |= condition_columns(p[k])
            if t in ("columns_added",):
                new |= set(p["target"])
            elif t == "column_value_relation":
                new.add(p["target"])
                need |= {a["col"] for a in p.get("args", []) if "col" in a}
            elif t in ("column_dtype", "null_count"):
                dtype_targets.add(p["target"])     # may refer to a new column
            elif p.get("target") not in (None, "*"):
                need |= set([p["target"]] if isinstance(p["target"], str) else p["target"])
            need |= set(p.get("except") or [])
    need |= dtype_targets - new
    return need, new
