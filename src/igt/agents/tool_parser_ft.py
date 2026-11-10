"""Tool-using FINE-TUNED parser (extension #13, prompt version `ft-t1`; docs/extensions.md).

The small fine-tuned parsers (1.5B, 3B) train and parse zero-shot with p1-v2. ft-t1 = p1-v2 + the tool instructions
of the tool-using parser (igt.agents.tool_parser.TOOL_ADDENDUM), zero-shot, the same read-only tools without rounding
(DataToolsFT) and the same
protocol: a tool call is a reply that is exactly one JSON object {"tool": ..., "args": {...}}; the final answer is the
contract in a ```json block. igt.agents.tool_parser (t1-v1, the 7B few-shot experiment) is unchanged.

    msgs = build_ft_tool_messages(task, schema)                      # system + one user turn (no few-shot examples)
    res = parse_with_tools_ft(task, schema, before_df, client)       # same result dict as tool_parser.parse_with_tools
"""
import hashlib
import json

from igt.agents.intent_parser import FT_PROMPT_VERSION, PROMPTS, USER_TEMPLATE, build_messages
from igt.agents.tool_parser import LAST, MAX_TOOL_CALLS, NEXT, TOOL_ADDENDUM, DataTools, tool_call_of

FT_TOOL_PROMPT_VERSION = "ft-t1"
SYSTEM_PROMPT_FT_TOOLS = PROMPTS[FT_PROMPT_VERSION] + TOOL_ADDENDUM
PROMPT_HASH_FT_TOOLS = hashlib.sha256((FT_TOOL_PROMPT_VERSION + SYSTEM_PROMPT_FT_TOOLS + USER_TEMPLATE)
                                      .encode()).hexdigest()[:12]
TOOL_RESULT_PREFIX = "Tool result:\n"
MAX_RESULT_CHARS = 4000


class DataToolsFT(DataTools):
    """The t1-v1 tools, except that numbers are not rounded. t1-v1 rounds floats to 6 decimals, so a median such as
    182.2569603 is shown as 182.25696 and a contract that copies it is off by more than the verifier's atol (1e-9).
    ft-t1 trains on tool results and contracts that agree exactly, so it shows full precision."""

    def profile_column(self, column, where=None):
        out = super().profile_column(column, where)
        from igt.agents.tool_parser import MAX_ROWS, _rows
        s = _rows(self.df, where)[column]
        out["top_values"] = [dict(value=k.item() if hasattr(k, "item") else k, count=int(v))
                             for k, v in s.value_counts(dropna=True).head(MAX_ROWS).items()]
        if all(k in out for k in ("min", "max", "mean", "median")):
            d = s.dropna()
            for k, v in (("min", d.min()), ("max", d.max()), ("mean", d.mean()), ("median", d.median())):
                out[k] = v.item() if hasattr(v, "item") else v
        return out


def build_ft_tool_messages(task, schema):
    msgs = build_messages(task, schema, [], FT_PROMPT_VERSION)
    msgs[0] = dict(role="system", content=SYSTEM_PROMPT_FT_TOOLS)
    return msgs


def tool_call_text(name, args):
    """The exact assistant text of a tool call in the training data (and the canonical form at parse time)."""
    return json.dumps({"tool": name, "args": args}, ensure_ascii=False)


def tool_result_text(result, calls_so_far, max_tool_calls=MAX_TOOL_CALLS):
    return TOOL_RESULT_PREFIX + json.dumps(result, default=str)[:MAX_RESULT_CHARS] + "\n\n" + \
        (NEXT if calls_so_far < max_tool_calls else LAST)


def parse_with_tools_ft(task, schema, before, client, max_tool_calls=MAX_TOOL_CALLS, max_new_tokens=1024):
    """-> dict(raw_output, tool_calls=[{tool, args, ok, result}], llm_calls, tokens, forced_answer, messages)."""
    tools = DataToolsFT(before)
    msgs = build_ft_tool_messages(task, schema)
    calls, forced = [], False
    while True:
        r = client.generate(msgs, max_new_tokens=max_new_tokens, temperature=0.0)
        tc = tool_call_of(r.text)
        done = tc is None or (len(calls) >= max_tool_calls and forced)
        if done:
            return dict(raw_output=r.text, tool_calls=calls, llm_calls=client.n_calls, tokens=client.tokens_used,
                        forced_answer=forced, messages=msgs + [dict(role="assistant", content=r.text)])
        if len(calls) >= max_tool_calls:
            forced = True
            msgs = msgs + [dict(role="assistant", content=r.text), dict(role="user", content=LAST)]
            continue
        ok, res = tools.call(*tc)
        calls.append(dict(tool=tc[0], args=tc[1], ok=ok, result=res))
        msgs = msgs + [dict(role="assistant", content=r.text),
                       dict(role="user", content=tool_result_text(res, len(calls), max_tool_calls))]
