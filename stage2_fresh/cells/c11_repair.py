# ===== Repair agent (Qwen2.5-3B-Instruct, FP16) with inspection tools =====
# Bounded loop: diagnose -> (optional tool calls) -> corrected code -> re-execute -> re-verify,
# at most MAX_REPAIR_ATTEMPTS times. The agent never decides acceptance: only the verifier does.

REPAIR_SYSTEM = ("You are the repair agent of a data-transformation system. Code written for a request was executed in a sandbox "
                 "and checked by a deterministic verifier against the intent contract, and it failed. Find the cause and return a "
                 "corrected complete function. You may call the inspection tools first. Your final answer is one short diagnosis "
                 "sentence followed by the complete corrected function in one ```python block.\n\n" + CODE_RULES)

REPAIR_TOOLS = [
    {"type": "function", "function": {
        "name": "inspect_rows",
        "description": "Show the input (before) and current output (after) values of some rows. Removed rows are marked.",
        "parameters": {"type": "object", "properties": {
            "row_ids": {"type": "array", "items": {"type": "integer"}, "description": "up to 10 row ids (index labels)"},
            "columns": {"type": "array", "items": {"type": "string"}, "description": "up to 8 column names"}},
            "required": ["row_ids", "columns"]}}},
    {"type": "function", "function": {
        "name": "column_summary",
        "description": "Compare one column in the input and in the current output: dtype, missing count, distinct values, top values, numeric statistics.",
        "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}}},
]


def _col_summary(s):
    vc = s.dropna().value_counts()
    out = {"dtype": str(s.dtype), "rows": int(len(s)), "missing": int(s.isna().sum()), "distinct": int(len(vc)),
           "top_values": [[_fmt(v), int(n)] for v, n in vc.head(5).items()]}
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().any():
        out.update(min=_fmt(s.min()), max=_fmt(s.max()), mean=float(s.mean()), median=float(s.median()))
    return out


def repair_tool(name, args, before, after):
    try:
        if name == "inspect_rows":
            rows = [r for r in args.get("row_ids", [])][:10]
            cols = [c for c in args.get("columns", [])][:8]
            res = []
            for r in rows:
                item = {"row": r}
                for c in cols:
                    item[f"before.{c}"] = _fmt(before.at[r, c]) if (r in before.index and c in before.columns) else "<no such row/column>"
                    if after is None:
                        item[f"after.{c}"] = "<no output>"
                    elif r not in after.index:
                        item[f"after.{c}"] = "<row removed>"
                    else:
                        item[f"after.{c}"] = _fmt(after.at[r, c]) if c in after.columns else "<column missing>"
                res.append(item)
            return {"ok": True, "rows": res}
        if name == "column_summary":
            c = args.get("column")
            return {"ok": True, "column": c,
                    "before": _col_summary(before[c]) if c in before.columns else "<not in input>",
                    "after": (_col_summary(after[c]) if (after is not None and c in after.columns) else "<not in output>")}
        return {"ok": False, "error": f"unknown tool {name!r}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def parse_tool_calls(text):
    calls = []
    for blk in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S):
        try:
            calls.append(json.loads(blk))
        except Exception:
            pass
    if not calls and "```python" not in text:
        calls = [o for o in extract_json_objects(text) if isinstance(o, dict) and "name" in o and "arguments" in o]
    return calls


def repair(task, before, contract, code, first_stage, first_output):
    user = ("Input data (variable `df`):\n" + describe_input_for_coder(before) +
            f"\n\nRequest: {task['request']}\n\nIntent contract:\n{contract_block(contract)}\n\n"
            f"Current code:\n```python\n{code or '# (no code)'}\n```\n\nResult of executing and verifying it:\n{first_stage['feedback']}\n\n"
            f"Diagnose the cause, then return the complete corrected function in one ```python block. "
            f"If you need evidence first, call a tool (at most {MAX_REPAIR_TOOL_CALLS} calls per attempt).")
    messages = [{"role": "system", "content": REPAIR_SYSTEM}, {"role": "user", "content": user}]
    attempts, calls, after = [], [], first_output
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        used_tools = []
        while True:
            text, usage = CODER_LM.chat(messages, max_new_tokens=REPAIR_MAX_NEW_TOKENS, tools=REPAIR_TOOLS)
            calls.append(usage)
            messages.append({"role": "assistant", "content": text})
            new_code = extract_code(text)
            tcs = parse_tool_calls(text)
            if new_code is None and tcs and len(used_tools) < MAX_REPAIR_TOOL_CALLS:
                for tc in tcs[: MAX_REPAIR_TOOL_CALLS - len(used_tools)]:
                    res = repair_tool(tc.get("name"), tc.get("arguments") or {}, before, after)
                    used_tools.append({"name": tc.get("name"), "arguments": tc.get("arguments")})
                    messages.append({"role": "tool", "content": json.dumps(res, default=str)})
                continue
            break
        stage, out = execute_and_verify(new_code, before, contract)
        stage["attempt"], stage["tools"] = attempt, used_tools
        attempts.append(stage)
        if out is not None:
            after = out
        if stage["status"] in ("PASS", "UNCERTAIN"):
            return attempts, calls, messages, out
        messages.append({"role": "user", "content": f"Attempt {attempt} was executed and verified again:\n{stage['feedback']}\n\n"
                                                    "Fix the remaining problem and return the complete corrected function in one ```python block."})
    return attempts, calls, messages, out  # output of the last attempt (None if it did not execute)
