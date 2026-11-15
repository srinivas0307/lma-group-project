# ===== Transformation planner + code generator (Qwen2.5-3B-Instruct, FP16) =====
CODE_RULES = """Rules:
1. Define `def transform(df):` that returns a new pandas DataFrame. Start with `out = df.copy()`.
2. `pd` (pandas) and `np` (numpy) are available; you may also import re, math, datetime. No other imports, no files, network, system calls or printing.
3. Keep the original index labels: never call reset_index, never use ignore_index=True, never sort or rebuild the frame. When rows are removed, the remaining rows keep their index labels.
4. Change only what the request asks for. Every other row, column, value and dtype stays exactly as it is. Do not leave helper columns in the output.
5. The output is checked automatically against the intent contract (what must change and what must be preserved). Satisfy every predicate.
6. Never convert, reformat or recompute columns the request does not mention (keep their dtype exactly).
7. Write only the function: no code outside it (no `with` blocks or statements around it), no assert statements, no printing or self-checks. The verifier does the checking.

pandas patterns to use:
- Remove duplicates: `out = df.drop_duplicates(subset=[key columns], keep="first")` (or keep="last"). Do NOT use groupby, merge, first()/last() or reset_index for deduplication.
- Filter rows: build a boolean mask and use `out = df.loc[mask].copy()`. "Remove rows where A and B" is `df.loc[~((A) & (B))]`; combine with `&`, `|`, `~` and parentheses.
- Fill missing values: `out[c] = out[c].fillna(value)`; only in some rows: `out.loc[mask & out[c].isna(), c] = value`.
- Rename: `out = out.rename(columns={"old": "new"})`; after this the old name no longer exists, do not access or drop it.
- Cap values: `out[c] = out[c].clip(lower=..., upper=...)`.
- Types: `out[c] = out[c].astype("float")`, `out[c] = pd.to_datetime(out[c], format=...)`.
- Categorical column: `out[c] = out[c].astype("category")` (unordered). ONLY when the request asks for an order: `out[c] = pd.Categorical(out[c], categories=[values in the requested order], ordered=True)`; never put missing values in the categories.
- Text values: `out[c] = out[c].str.lower()` (missing values stay missing; do not call .lower() on every value with apply)."""

GEN_SYSTEM = "You are a data engineer who writes pandas code. You implement exactly one requested data transformation as a Python function.\n\n" + CODE_RULES


def describe_input_for_coder(df, n_rows=5):
    text = describe_input(df).replace("First 3 rows (CSV):\n" + df.head(3).to_csv(index=False), "")
    text += f"First {n_rows} rows (CSV, index = row id):\n" + df.head(n_rows).to_csv()
    small = {c: sorted(map(str, df[c].dropna().unique()))[:20] for c in df.columns
             if df[c].dtype == object and df[c].nunique() <= 20}
    if small:
        text += "\nValues of low-cardinality text columns:\n" + "\n".join(f"- {c}: {v}" for c, v in small.items())
    return text


def contract_block(contract):
    return describe_contract(contract) + "\n\nContract JSON:\n```json\n" + json.dumps(contract, indent=1, default=str) + "\n```"


def generate_code(task, df, contract):
    user = ("Input data (variable `df`):\n" + describe_input_for_coder(df) +
            f"\n\nRequest: {task['request']}\n\nIntent contract:\n{contract_block(contract)}\n\n"
            "First write a short plan (2-5 bullet points) as Python comments, then the complete function in one ```python block.")
    messages = [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}]
    text, usage = CODER_LM.chat(messages, max_new_tokens=GEN_MAX_NEW_TOKENS)
    messages.append({"role": "assistant", "content": text})
    calls, code = [usage], extract_code(text)
    problem = format_problem(code)
    if problem:   # one format retry (unusable reply is not a repair attempt)
        messages.append({"role": "user", "content": f"Your reply could not be used: {problem}. Return the complete function "
                                                     "`def transform(df):` (with `return out`) in one ```python block."})
        text, usage = CODER_LM.chat(messages, max_new_tokens=GEN_MAX_NEW_TOKENS)
        messages.append({"role": "assistant", "content": text})
        calls.append(usage)
        code = extract_code(text)
    return {"messages": messages, "calls": calls, "code": code, "format_retry": bool(problem)}
