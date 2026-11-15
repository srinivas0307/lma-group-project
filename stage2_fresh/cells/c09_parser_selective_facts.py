# ===== INTENT PARSER (variant A): fine-tuned 1.5B "selective facts" adapter (protocol p1-v2+sf1) =====
# Before the model runs, a deterministic interpreter computes ONLY the input statistics the request needs
# (median / mean / mode, full-column, one equality group, or one fact per group) and inserts them as
# "Required input facts (sf1)". The model then writes the contract in ONE reply. No tools.
# Feedback loop (added): schema errors, unknown columns/values, vacuous or uncheckable contracts are sent
# back to the parser for up to MAX_PARSER_FEEDBACK correction rounds.

PARSER_SYSTEM = __SYSTEM_PROMPT__


class FactUnavailable(Exception):
    pass


_STATS = [("median", r"\bmedian\b"), ("mean", r"\b(?:mean|average)\b"), ("mode", r"\b(?:mode|most (?:common|frequent))\b")]


def _col_spans(request, cols):
    """(start, column) for every input column mentioned in the request (case-insensitive, whole token)."""
    hits = []
    for c in cols:
        for form in {c, c.replace("_", " ")}:
            for m in re.finditer(r"(?<![\w-])`?" + re.escape(form) + r"`?(?![\w-])", request, flags=re.I):
                hits.append((m.start(), c))
    return sorted(set(hits))


def _match_value(series, raw):
    raw = raw.strip().strip("'\"`")
    vals = series.dropna().unique().tolist()
    for v in vals:
        if str(v) == raw:
            return _fmt(v)
    for v in vals:
        if str(v).lower() == raw.lower():
            return _fmt(v)
    try:
        f = float(raw)
        for v in vals:
            if _is_number(v) and float(v) == f:
                return _fmt(v)
    except ValueError:
        pass
    raise FactUnavailable(f"value {raw!r} not found in column {series.name!r}")


def _statistic(series, stat):
    s = series.dropna()
    if len(s) == 0:
        raise FactUnavailable(f"{series.name!r} has no non-missing values in scope")
    if stat in ("median", "mean"):
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            raise FactUnavailable(f"{stat} of non-numeric column {series.name!r}")
        return float(s.median() if stat == "median" else s.mean()), int(len(s))
    vc = s.value_counts()
    top = vc[vc == vc.iloc[0]]
    if len(top) > 1:
        raise FactUnavailable(f"mode of {series.name!r} is tied between {[_fmt(v) for v in top.index[:5]]}")
    return _fmt(top.index[0]), int(len(s))


def compute_facts(request, df):
    found = [s for s, pat in _STATS if re.search(pat, request, flags=re.I)]
    if not found:
        return []
    if len(found) > 1:
        raise FactUnavailable(f"several statistics requested: {found}")
    stat = found[0]
    stat_pat = dict(_STATS)[stat]
    cols = list(df.columns)
    spans = _col_spans(request, cols)
    if not spans:
        raise FactUnavailable("the request names no input column")
    target = None
    for c in sorted(cols, key=len, reverse=True):       # "median of X" / "median X" / "X's median"
        cp = r"`?" + re.escape(c) + r"`?(?![\w-])"
        if re.search(stat_pat + r"\s+(?:value\s+)?(?:of\s+)?(?:the\s+)?(?:column\s+)?" + cp, request, flags=re.I) \
                or re.search(r"(?<![\w-])" + cp + r"'s\s+" + stat_pat, request, flags=re.I):
            target = c
            break
    m_scope = re.search(r"(?:among|within|for|in)\s+(?:the\s+)?rows\s+(?:with|where)\s+`?([\w\- ]+?)`?\s*(?:==|=|is)\s*([^\s,.;]+)", request, flags=re.I)
    scope_col = None
    if m_scope:
        scope_col = next((c for c in cols if c.lower() == m_scope.group(1).strip().lower()), None)
        if scope_col is None:
            raise FactUnavailable(f"scope column {m_scope.group(1)!r} not in the input")
    if target is None:                                     # fall back to the filled column
        m_fill = re.search(r"missing\s+(?:values?\s+(?:in|of)\s+)?`?([\w\-]+)`?", request, flags=re.I)
        cand = [c for _, c in spans if c != scope_col]
        if m_fill and any(c.lower() == m_fill.group(1).lower() for c in cols):
            target = next(c for c in cols if c.lower() == m_fill.group(1).lower())
        elif len(set(cand)) == 1:
            target = cand[0]
        else:
            raise FactUnavailable(f"cannot tell which column the {stat} is of (mentioned: {sorted(set(cand))})")
    full = bool(re.search(r"full[- ]column|whole column|overall|entire column", request, flags=re.I))
    group_col = None
    if not m_scope and not full:
        for m_grp in re.finditer(r"\b(?:same|per|each|by)\b", request, flags=re.I):
            later = [c for pos, c in spans if pos > m_grp.start() and c != target]
            if later:
                group_col = later[0]
                break
    facts = []
    if m_scope:
        val = _match_value(df[scope_col], m_scope.group(2))
        where = {"column": scope_col, "op": "==", "value": val}
        v, n = _statistic(df.loc[df[scope_col] == val, target], stat)
        facts.append({"column": target, "statistic": stat, "value": v, "scope": {"kind": "conditional", "where": where},
                      "nonmissing_count": n, "missing_values": "excluded"})
    elif group_col:
        groups = df[group_col].dropna().unique().tolist()
        if len(groups) > 12:
            raise FactUnavailable(f"too many groups in {group_col!r} ({len(groups)})")
        for g in sorted(groups, key=lambda x: (str(type(x)), x)):
            g = _fmt(g)
            v, n = _statistic(df.loc[df[group_col] == g, target], stat)
            facts.append({"column": target, "statistic": stat, "value": v,
                          "scope": {"kind": "conditional", "where": {"column": group_col, "op": "==", "value": g}},
                          "nonmissing_count": n, "missing_values": "excluded"})
    else:
        v, n = _statistic(df[target], stat)
        facts.append({"column": target, "statistic": stat, "value": v, "scope": {"kind": "full_column"},
                      "nonmissing_count": n, "missing_values": "excluded"})
    return facts


def build_parser_user_message(request, df, facts):
    msg = describe_input(df)
    if facts:
        msg += "\nRequired input facts (sf1):\n" + json.dumps(facts) + "\n"
    return msg + "\nRequest: " + request


def _feedback_message(problems):
    return ("The contract was rejected by the contract checker:\n- " + "\n- ".join(problems) +
            "\nReturn the complete corrected contract as one ```json block and nothing else.")


def run_parser(task, df):
    rec = {"variant": "selective_facts", "calls": [], "feedback": [], "facts": None}
    try:
        facts = compute_facts(task["request"], df)
    except FactUnavailable as e:
        rec.update(status="fact_unavailable", error=str(e))
        return rec
    rec["facts"] = facts
    messages = [{"role": "system", "content": PARSER_SYSTEM},
                {"role": "user", "content": build_parser_user_message(task["request"], df, facts)}]
    contract, errors, hard, soft = None, [], [], []
    for rnd in range(MAX_PARSER_FEEDBACK + 1):
        text, usage = PARSER_LM.chat(messages, max_new_tokens=PARSER_MAX_NEW_TOKENS)
        rec["calls"].append(usage)
        messages.append({"role": "assistant", "content": text})
        objs = [o for o in extract_json_objects(text) if isinstance(o, dict)]
        cands = [o for o in objs if "must_change" in o or "must_preserve" in o]
        if any(o.get("action") == "escalate" for o in objs) and not cands:
            rec.update(status="escalated", error=next(o.get("reason") for o in objs if o.get("action") == "escalate"))
            break
        if not cands:
            contract, errors = None, ["no JSON contract found in the reply" + (" (the reply was cut off; write it more compactly)" if usage["truncated"] else "")]
        else:
            contract, errors = normalize_contract(cands[-1])
        hard, soft = contract_issues(contract, df) if contract else ([], [])
        problems = errors or (hard + soft)
        if not problems:
            break
        rec["feedback"].append({"round": rnd + 1, "problems": problems})
        if rnd == MAX_PARSER_FEEDBACK:
            break
        messages.append({"role": "user", "content": _feedback_message(problems)})
    rec["messages"] = messages
    if rec.get("status") == "escalated":
        return rec
    if errors:
        rec.update(status="parser_invalid", error="; ".join(errors))
    elif hard:
        rec.update(status="parser_rejected", error="; ".join(hard), contract=contract)
    else:
        rec.update(status="ok", contract=contract, warnings=soft)
    return rec
