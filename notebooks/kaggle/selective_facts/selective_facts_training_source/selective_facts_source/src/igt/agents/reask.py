"""Re-ask once on an invalid contract (extension #8, version `rq1`), optionally after the data pre-check (#9, `pc1rq1`).

    msgs = original_messages(parse_row, task, schema, examples)       # the parser's conversation, rebuilt
    msgs += reask_turns(parse_row["raw_output"], problem_text)        # its answer + "Your contract is invalid: ..."
    second = client.generate(msgs, ...)                               # the second answer replaces the first

Which conversation is rebuilt depends on how the row was parsed:
    prompted (p1-v1..v3, few- or zero-shot)   build_messages(task, schema, examples | [], version)   exact
    fine-tuned (adapter, zero-shot p1-v2)     build_messages(task, schema, [], version)              exact
    tool-using t1-v1 / ft-t1                  system + user + one (tool call, tool result) pair per recorded call;
                                              the tool-call replies are re-serialized from the recorded calls
                                              (approximate when the model wrote extra text around the JSON)
"""
import json

from igt.agents.intent_parser import build_messages

REASK_VERSION = "rq1"
REASK_INVALID = ("Your contract is invalid: {error}\n"
                 "Reply with a corrected contract in a ```json block, using only the vocabulary described above.")
REASK_PRECHECK = ("Your contract does not fit the input data:\n{issues}\n"
                  "Reply with a corrected contract in a ```json block, using only the vocabulary described above. "
                  "Use actual values from the data, not names of statistics.")


def parser_kind(row):
    v = row.get("parser_prompt_version")
    if v == "t1-v1":
        return "tools"
    if v == "ft-t1":
        return "tools_ft"
    return "ft" if row.get("adapter") else "prompted"


def original_messages(row, task, schema, examples):
    kind = parser_kind(row)
    if kind in ("prompted", "ft"):
        ex = examples if (kind == "prompted" and row.get("shots", "few") == "few") else []
        return build_messages(task, schema, ex, row["parser_prompt_version"])
    if kind == "tools":
        from igt.agents.tool_parser import build_tool_messages
        msgs = build_tool_messages(task, schema, examples)
    else:
        from igt.agents.tool_parser_ft import build_ft_tool_messages
        msgs = build_ft_tool_messages(task, schema)
    from igt.agents.tool_parser import LAST, MAX_TOOL_CALLS, NEXT
    calls = row.get("tool_calls") or []
    for i, c in enumerate(calls, 1):
        msgs = msgs + [dict(role="assistant", content=json.dumps({"tool": c["tool"], "args": c["args"]})),
                       dict(role="user", content="Tool result:\n" + json.dumps(c.get("result"), default=str)[:4000] +
                            "\n\n" + (NEXT if i < MAX_TOOL_CALLS else LAST))]
    return msgs


def reask_turns(first_output, problem):
    """problem: ('invalid', error text) or ('precheck', issues text)."""
    kind, text = problem
    msg = REASK_INVALID.format(error=text) if kind == "invalid" else REASK_PRECHECK.format(issues=text)
    return [dict(role="assistant", content=first_output), dict(role="user", content=msg)]
