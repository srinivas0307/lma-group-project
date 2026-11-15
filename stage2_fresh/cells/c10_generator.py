# ===== Transformation planner + code generator (Qwen2.5-3B-Instruct, FP16) =====
CODE_RULES = """Rules:
1. Define `def transform(df):` that returns a new pandas DataFrame. Start with `out = df.copy()`.
2. `pd` (pandas) and `np` (numpy) are available; you may also import re, math, datetime. No other imports, no files, network, system calls or printing.
3. Keep the original index labels: never call reset_index, never use ignore_index=True, never sort or rebuild the frame. When rows are removed, the remaining rows keep their index labels.
4. Change only what the request asks for. Every other row, column, value and dtype stays exactly as it is. Do not leave helper columns in the output.
5. The output is checked automatically against the intent contract (what must change and what must be preserved). Satisfy every predicate."""

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
    return {"messages": messages + [{"role": "assistant", "content": text}], "calls": [usage], "code": extract_code(text)}
