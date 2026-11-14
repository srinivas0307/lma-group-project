"""Tool-using intent parser (extra experiment, README §12 2026-10-31): the parser may look at the full input data with
read-only tools before it writes the contract. Everything else is Baseline 3: the same base prompt (p1-v3), the same
5 dev few-shot examples, greedy decoding, the same JSON extraction and strict validation.

    from igt.agents.tool_parser import parse_with_tools
    res = parse_with_tools(task, schema, examples, before_df, client)    # res["raw_output"]: the final reply

Why: some requests need a value that only the data can answer (a median, the most common value, exact category
spellings, whether a column has missing values). Baseline 3 sees only the schema and 3 rows; on titanic_miss_004 and
airbnb_miss_004 it wrote the text "median" as the fill value.

Tools (read-only, on the BEFORE data; no code execution, nothing about outputs, references or contracts):
  profile_column(column, where=COND?)   dtype, rows, missing, distinct, top values, min/max/mean/median, on the rows
                                        matching the optional condition (same condition grammar as the contract)
  count_rows(where=COND)                number of input rows matching a condition
  inspect_rows(columns?, where=COND?, n?) up to 10 matching rows
Protocol: a tool call is a reply that is exactly one JSON object {"tool": ..., "args": {...}}; the final answer is the
contract in a ```json block, as in Baseline 3. At most MAX_TOOL_CALLS calls, then the model must answer.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from igt.agents.intent_parser import PROMPTS, USER_TEMPLATE, FEW_SHOT_IDS, _first_json_object, build_messages

TOOL_PROMPT_VERSION = "t1-v1"
BASE_PROMPT_VERSION = "p1-v3"
MAX_TOOL_CALLS = 6
MAX_ROWS = 10

TOOL_ADDENDUM = """

Tools: before answering you may look at the full input data. Use a tool when the contract needs a value from the data
(a median or mean, the most common value, the exact spelling of category values, whether a column has missing
values, how many rows match a condition). Don't use tools when the request already gives every value.
To call a tool, reply with exactly one JSON object and nothing else:
  {"tool": "profile_column", "args": {"column": "score"}}
  {"tool": "profile_column", "args": {"column": "score", "where": {"column": "group", "op": "==", "value": "A"}}}
  {"tool": "count_rows", "args": {"where": {"column": "score", "op": "<", "value": 10}}}
  {"tool": "inspect_rows", "args": {"columns": ["group", "score"], "where": {"column": "score", "op": "is_null"}, "n": 5}}
profile_column returns dtype, rows, missing, distinct count, the most common values and, for numbers, min, max, mean
and median, computed on the rows matching "where" (if given). "where" uses the COND format above.
After each result, call another tool or give your final answer: the contract in a ```json block. At most %d tool
calls. Values you take from the data go into the contract as literals: {"lit": 28.0} or a condition "value".""" % MAX_TOOL_CALLS

SYSTEM_PROMPT = PROMPTS[BASE_PROMPT_VERSION] + TOOL_ADDENDUM
PROMPT_HASH = hashlib.sha256((TOOL_PROMPT_VERSION + SYSTEM_PROMPT + USER_TEMPLATE + ",".join(FEW_SHOT_IDS))
                             .encode()).hexdigest()[:12]
NEXT = "Call another tool, or give the final contract in a ```json block."
LAST = "No more tool calls. Give the final contract now, in a ```json block."


def _py(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v if isinstance(v, (str, int, float, bool)) else str(v)


def _rows(df, where):
    if where is None:
        return df
    from igt.contracts.schema import Condition
    from igt.verifier.conditions import evaluate
    return df[evaluate(Condition.model_validate(where), df)]


class DataTools:
    """Read-only tools over one input DataFrame. call(name, args) -> (ok, result dict)."""

    def __init__(self, df):
        self.df = df

    def profile_column(self, column, where=None):
        if column not in self.df.columns:
            raise KeyError(f"no column '{column}'; columns: {[str(c) for c in self.df.columns]}")
        s = _rows(self.df, where)[column]
        out = dict(column=column, dtype=str(s.dtype), rows=len(s), missing=int(s.isna().sum()),
                   distinct=int(s.nunique(dropna=True)),
                   top_values=[dict(value=_py(k), count=int(v)) for k, v in s.value_counts(dropna=True).head(MAX_ROWS).items()])
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().any():
            d = s.dropna()
            out.update(min=_py(d.min()), max=_py(d.max()), mean=_py(d.mean()), median=_py(d.median()))
        return out

    def count_rows(self, where):
        return dict(matching_rows=len(_rows(self.df, where)), total_rows=len(self.df))

    def inspect_rows(self, columns=None, where=None, n=5):
        cols = columns or [str(c) for c in self.df.columns]
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise KeyError(f"no columns {missing}")
        sub = _rows(self.df, where)
        return dict(matching_rows=len(sub),
                    rows=[{str(k): _py(v) for k, v in r.items()} for _, r in sub[cols].head(min(int(n), MAX_ROWS)).iterrows()])

    def call(self, name, args):
        fn = {"profile_column": self.profile_column, "count_rows": self.count_rows,
              "inspect_rows": self.inspect_rows}.get(name)
        if fn is None:
            return False, dict(error=f"unknown tool '{name}'; tools: profile_column, count_rows, inspect_rows")
        if not isinstance(args, dict):
            return False, dict(error='"args" must be a JSON object')
        try:
            return True, fn(**args)
        except TypeError as e:
            return False, dict(error=f"bad arguments for {name}: {e}")
        except Exception as e:                        # unknown column, invalid condition, ...
            return False, dict(error=f"{type(e).__name__}: {str(e)[:300]}")


def tool_call_of(text):
    """A reply that is a tool call -> (name, args); a final answer or anything else -> None."""
    raw = _first_json_object(text)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("tool"), str) and "must_change" not in obj and "must_preserve" not in obj:
        return obj["tool"], obj.get("args", {})
    return None


def build_tool_messages(task, schema, examples):
    msgs = build_messages(task, schema, examples, BASE_PROMPT_VERSION)
    msgs[0] = dict(role="system", content=SYSTEM_PROMPT)
    return msgs


def parse_with_tools(task, schema, examples, before, client, max_tool_calls=MAX_TOOL_CALLS, max_new_tokens=1024):
    """-> dict(raw_output, tool_calls=[{tool, args, ok, result}], llm_calls, tokens, forced_answer)."""
    tools = DataTools(before)
    msgs = build_tool_messages(task, schema, examples)
    calls, forced = [], False
    while True:
        r = client.generate(msgs, max_new_tokens=max_new_tokens, temperature=0.0)
        tc = tool_call_of(r.text)
        if tc is None:
            return dict(raw_output=r.text, tool_calls=calls, llm_calls=client.n_calls, tokens=client.tokens_used,
                        forced_answer=forced)
        if len(calls) >= max_tool_calls:
            if forced:                                 # asked to answer and still calling tools: stop here
                return dict(raw_output=r.text, tool_calls=calls, llm_calls=client.n_calls, tokens=client.tokens_used,
                            forced_answer=True)
            forced = True
            msgs += [dict(role="assistant", content=r.text), dict(role="user", content=LAST)]
            continue
        ok, res = tools.call(*tc)
        calls.append(dict(tool=tc[0], args=tc[1], ok=ok, result=res))
        msgs += [dict(role="assistant", content=r.text),
                 dict(role="user", content="Tool result:\n" + json.dumps(res, default=str)[:4000] + "\n\n" +
                      (NEXT if len(calls) < max_tool_calls else LAST))]
