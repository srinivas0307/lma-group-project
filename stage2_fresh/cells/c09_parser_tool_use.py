# ===== INTENT PARSER (variant B): fine-tuned 1.5B "tool use" adapter (protocol parser-tools-v2) =====
# The model sees the schema + 3 rows and DECIDES itself whether to query the input data with read-only
# tools (profile_column / count_rows / inspect_rows / group_profile) before writing the contract.
# Multi-turn protocol exactly as trained: tool call -> "Read-only input-data tool result" -> ... -> contract
# or {"action": "escalate"}. Feedback loop: schema errors use the trained "Invalid response ..." correction
# message; the added input-data checks (unknown columns/values, vacuous/uncheckable contract) use the same channel.

PARSER_SYSTEM = __SYSTEM_PROMPT__

MAX_TOOL_ATTEMPTS, MAX_REPLIES, MAX_PARSER_TOKENS = 6, 10, 24000
NULL_POLICY = "Statistics and frequencies exclude missing cells; no tie-break is inferred."
CORRECTION = ('Invalid response. Send one tool-call JSON object, a schema-valid final contract in a json fence, '
              'or {"action":"escalate","reason":"..."}. Parser error: ')


class DataTools:
    """Read-only queries over the INPUT DataFrame only (no outputs, no reference, no code execution)."""

    def __init__(self, df):
        self.df = df

    def _col(self, c):
        if not isinstance(c, str) or c not in self.df.columns:
            raise ValueError(f"Unknown input column: {c!r}")
        return c

    def _rows(self, where):
        if where is None:
            return self.df
        errs = []
        validate_condition(where, "where", errs)
        if errs:
            raise ValueError("; ".join(errs))
        for c in condition_columns(where):
            self._col(c)
        return self.df[eval_condition(self.df, where).values]

    @staticmethod
    def _profile(s):
        nonnull = s.dropna()
        vc = nonnull.value_counts()
        top = [{"value": _fmt(v), "count": int(n)} for v, n in vc.head(10).items()]
        modes_all = sorted([_fmt(v) for v in vc[vc == vc.iloc[0]].index], key=lambda x: (str(type(x)), x)) if len(vc) else []
        out = {"rows": int(len(s)), "missing": int(s.isna().sum()), "dtype": str(s.dtype), "distinct": int(len(vc)),
               "top_values": top, "values_omitted": int(max(0, len(vc) - 10)), "modes": modes_all[:10],
               "mode_tie": len(modes_all) > 1, "mode_values_total": len(modes_all), "modes_truncated": len(modes_all) > 10,
               "null_policy": NULL_POLICY}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and len(nonnull):
            f = nonnull.astype(float)
            fin = f[np.isfinite(f)]
            out["statistics"] = {"min": _fmt(fin.min()), "max": _fmt(fin.max()), "mean": float(fin.mean()), "median": float(fin.median())}
            out["nonfinite_nonmissing"] = int((~np.isfinite(f)).sum())
        return out

    def profile_column(self, column, where=None):
        c = self._col(column)
        return {"column": c, "where": where, **self._profile(self._rows(where)[c])}

    def count_rows(self, where):
        return {"where": where, "rows": int(len(self.df)), "matching": int(len(self._rows(where)))}

    def inspect_rows(self, columns, n=3, where=None):
        if not isinstance(columns, list) or not 1 <= len(columns) <= 8:
            raise ValueError("columns must be a list of 1 to 8 input columns")
        if not isinstance(n, int) or not 1 <= n <= 5:
            raise ValueError("n must be an integer from 1 to 5")
        cols = [self._col(c) for c in columns]
        rows = self._rows(where)
        return {"columns": cols, "where": where, "matching": int(len(rows)),
                "rows": [{"row": int(i), **{c: _fmt(rows.at[i, c]) for c in cols}} for i in rows.index[:n]]}

    def group_profile(self, column, by, where=None):
        c = self._col(column)
        if not isinstance(by, list) or not 1 <= len(by) <= 2:
            raise ValueError("by must contain 1 or 2 input columns")
        by = [self._col(b) for b in by]
        rows = self._rows(where)
        groups = []
        for key, g in rows.groupby(by, dropna=False, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            p = self._profile(g[c])
            p.pop("null_policy"); p.pop("top_values"); p.pop("values_omitted")
            groups.append({"key": {b: _fmt(k) for b, k in zip(by, key)}, **p})
        return {"column": c, "by": by, "where": where, "groups_total": len(groups), "truncated": len(groups) > 12,
                "groups": groups[:12], "null_policy": NULL_POLICY}

    def call(self, name, args):
        if name not in ("profile_column", "count_rows", "inspect_rows", "group_profile"):
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(args, dict):
            raise ValueError("args must be an object")
        return getattr(self, name)(**args)


def _observation(obs):
    return "Read-only input-data tool result:\n" + json.dumps(obs, default=str) + "\nChoose another tool, return the final contract, or escalate."


def run_parser(task, df):
    rec = {"variant": "tool_use", "calls": [], "feedback": [], "tool_calls": []}
    tools = DataTools(df)
    messages = [{"role": "system", "content": PARSER_SYSTEM},
                {"role": "user", "content": describe_input(df) + "\nRequest: " + task["request"]}]
    corrections, contract, errors, hard, soft = 0, None, [], [], []
    status = None
    for reply_no in range(MAX_REPLIES):
        if tokens_of(rec["calls"]) > MAX_PARSER_TOKENS:
            status, rec["error"] = "parser_budget", f"cumulative parser tokens exceeded {MAX_PARSER_TOKENS}"
            break
        text, usage = PARSER_LM.chat(messages, max_new_tokens=PARSER_MAX_NEW_TOKENS)
        rec["calls"].append(usage)
        messages.append({"role": "assistant", "content": text})
        objs = [o for o in extract_json_objects(text) if isinstance(o, dict)]
        first = objs[0] if objs else None
        if first is not None and first.get("action") == "escalate":
            status, rec["error"] = "escalated", str(first.get("reason"))
            break
        if first is not None and "tool" in first and "must_change" not in first:
            ok_budget = len(rec["tool_calls"]) < MAX_TOOL_ATTEMPTS
            try:
                if not ok_budget:
                    raise RuntimeError("tool attempt budget exhausted; return the final contract or escalate")
                obs = {"ok": True, "result": json.loads(json.dumps(tools.call(first["tool"], first.get("args", {})), default=_fmt))}
            except Exception as e:
                obs = {"ok": False, "error_type": type(e).__name__, "error": str(e)}
            rec["tool_calls"].append({"tool": first.get("tool"), "args": first.get("args"), "observation": obs})
            messages.append({"role": "user", "content": _observation(obs)})
            continue
        cands = [o for o in objs if "must_change" in o or "must_preserve" in o]
        if not cands:
            contract, errors = None, ["no tool call, contract or escalation JSON found" + (" (reply was cut off)" if usage["truncated"] else "")]
        else:
            contract, errors = normalize_contract(cands[-1])
        hard, soft = contract_issues(contract, df) if contract else ([], [])
        if errors:
            problem = "invalid contract: " + "; ".join(errors)
        elif hard or soft:
            problem = "contract rejected by input-data checks: " + "; ".join(hard + soft)
        else:
            status = "ok"
            break
        rec["feedback"].append({"round": corrections + 1, "problems": errors or (hard + soft)})
        if corrections >= MAX_PARSER_FEEDBACK:
            break
        corrections += 1
        messages.append({"role": "user", "content": CORRECTION + problem})
    rec["messages"] = messages
    rec["tool_attempts"] = len(rec["tool_calls"])
    rec["tool_failures"] = sum(not c["observation"]["ok"] for c in rec["tool_calls"])
    if status in ("escalated", "parser_budget"):
        rec["status"] = status
    elif status == "ok":
        rec.update(status="ok", contract=contract, warnings=[])
    elif errors or contract is None:
        rec.update(status="parser_invalid", error="; ".join(errors) or "reply budget exhausted")
    elif hard:
        rec.update(status="parser_rejected", error="; ".join(hard), contract=contract)
    else:
        rec.update(status="ok", contract=contract, warnings=soft)
    return rec
