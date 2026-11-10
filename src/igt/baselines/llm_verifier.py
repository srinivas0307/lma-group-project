"""LLM-as-verifier baseline (extension #3, prompt version `lv1`; docs/extensions.md). No contract: a model judges each
candidate from the request, the input schema and a summary of what the code changed.

    msgs = build_messages(request, schema_text, before, after)
    verdict, info = judge_reply(reply_text)          # Verdict + dict(reason, malformed)

The change summary comes from the agent's diff tool (igt.tools diff_datasets: rows removed / added, changed cells per
column with examples, dtype changes, new and dropped columns), plus the output's columns and first rows.
Verdict format v1.0 (docs/verdict_format.md) without changing it: verifier slot "conventional" (a checker without a
contract), REJECT = VIOLATION with one check failure {check: "llm_judge", value: <reason>, threshold: null},
a malformed reply = UNCERTAIN (reason predicate_not_evaluable) = ESCALATE. meta.verifier_kind = "llm".
"""
import hashlib
import json

import pandas as pd

from igt.agents.intent_parser import _first_json_object
from igt.types import Verdict, check_failure, uncertain_failure

LLM_VERIFIER_VERSION = "lv1"
VERIFIER_SLOT = "conventional"

SYSTEM = """You check the output of a data transformation. A user asked for a change to a table; some code was run on the table. You see the request, the input table's schema, and a summary of exactly what the code changed.

Decide whether the output does exactly what the request asks:
- every change the request asks for was made, on the right rows and columns;
- nothing else changed: no other rows removed or added, no other cells or columns changed, no column types changed unless the request asks for it.

Reply with one JSON object and nothing else:
{"verdict": "ACCEPT", "reason": "<one sentence>"}
or
{"verdict": "REJECT", "reason": "<one sentence naming what is wrong>"}"""

USER = """Request: {request}

Input data:
{schema}

What the code changed (input -> output):
{diff}

Output columns: {out_cols}
First {k} output rows (CSV):
{out_head}

Does the output do exactly what the request asks? Reply with the JSON object only."""

PROMPT_HASH = hashlib.sha256((LLM_VERIFIER_VERSION + SYSTEM + USER).encode()).hexdigest()[:12]
MAX_DIFF_CHARS = 5000


def change_summary(before, after, max_examples=3):
    from igt.tools import ToolCall, ToolContext, build_registry
    ctx = ToolContext(before=before, verify_fn=lambda a: None)
    ctx.last_output = after
    r = build_registry(ctx).call(ToolCall("diff_datasets", {"max_examples": max_examples}))
    if not r.ok:
        return dict(error=str(r.error))
    return r.output


def build_messages(request, schema_text, before, after, k=3):
    diff = json.dumps(change_summary(before, after), default=str)[:MAX_DIFF_CHARS]
    out_cols = ", ".join(f"{c} ({after[c].dtype})" for c in after.columns)
    out_head = after.head(k).to_csv(index=False) if isinstance(after, pd.DataFrame) else ""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(request=request, schema=schema_text, diff=diff, out_cols=out_cols,
                                                    k=k, out_head=out_head)}]


def judge_reply(text):
    raw = _first_json_object(text or "")
    try:
        obj = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        obj = None
    v = str(obj.get("verdict", "")).strip().upper() if isinstance(obj, dict) else ""
    reason = str(obj.get("reason", ""))[:300] if isinstance(obj, dict) else ""
    if v == "ACCEPT":
        return Verdict("PASS", [], []), dict(reason=reason, malformed=False)
    if v == "REJECT":
        return Verdict.combine([check_failure("llm_judge", reason or "rejected", None)]), dict(reason=reason, malformed=False)
    return Verdict.combine([uncertain_failure("predicate_not_evaluable", "malformed LLM verdict: " + (text or "")[:200])]), \
        dict(reason=None, malformed=True)
