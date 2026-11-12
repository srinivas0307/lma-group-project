"""Opt-in a1-v5 prompt; the a1-v4 loop, tools and acceptance policy are unchanged."""
import hashlib

from . import repair_agent as original

AGENT_PROMPT_VERSION = "a1-v5"
REPAIR_RULES = """

Repair discipline:
- Diagnose the named failing check and its expected/observed values before choosing an edit. A dtype-preservation failure needs the unnecessary cast removed; rewriting the row filter does not fix it.
- Keep the parts that already satisfy their checks. Use exact column names and preserve unrelated data types, values and columns. Remove temporary helper columns before returning.
- Reason about the full Boolean condition: removing rows matching a conjunction means retaining the complement of that entire conjunction, not applying separate complementary filters.
- Use pandas 2.3.3 / numpy 2.1.3 APIs. Do not use DataFrame.append or a boolean inclusive argument to Series.between.
- A null and the literal string "missing" are different. Follow the request and the read-only contract; do not reinterpret an explicit null requirement as a text label.
- run_transformation tests a candidate; run_verifier checks that tested candidate. Neither tool submits it as your proposed fix.
- When the tested candidate passes run_verifier and satisfies the request, your next action should be propose_fix with that exact complete code. Passing a tool check is a reason to submit the candidate, not a reason to escalate. Submitted code is executed and checked again by the system.
- If a candidate fails, change the cause identified in the report. Do not repeat identical code or make cosmetic rewrites while leaving the same failed check unresolved.
- Escalate only for a concrete ambiguity, missing required data, incompatible requirements, or an unsafe request. Do not escalate merely because the last tested candidate passed.
"""
PROMPT_HASH = hashlib.sha256((AGENT_PROMPT_VERSION + original.FULL_SYSTEM + REPAIR_RULES).encode()).hexdigest()[:12]


class _RunV5(original._Run):
    def full_messages(self, observations):
        messages = super().full_messages(observations)
        return [dict(messages[0], content=messages[0]["content"] + REPAIR_RULES), *messages[1:]]


def run_repair(task, initial_code, before, verify_fn, client, config=None, contract_text=None, schema_text=None):
    cfg = config or original.AgentConfig()
    if cfg.mode not in ("full", "none"):
        raise ValueError("a1-v5 changes the full-agent prompt; use full or none mode")
    if client.token_budget is None:
        client.token_budget = cfg.limits.token_budget
    return _RunV5(task, initial_code, before, verify_fn, client, cfg, contract_text, schema_text).run()
