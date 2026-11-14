"""Opt-in variants of the repair agent (docs/extensions.md). The a1-v4 agent in repair_agent.py is unchanged; these
variants subclass its run loop, and only eval/runner_agent_ext.py / eval/runner_stage2_ext.py use them.

    np1  stop after k consecutive repair attempts that make no progress (AgentConfigExt.stop_after_no_progress = k).
         "No progress" is the definition of eval/agent_metrics.no_progress: the attempt doesn't pass and its failed
         checks are not a strict subset of the previous attempt's (a crash, a block or a repeated code is no progress;
         code that runs after code that crashed is progress). The run ends as ESCALATE, reason "no progress ...".
         Path: ... FAILED -> REPAIR -> ESCALATED (the state machine allows ESCALATED only from REPAIR / PARSE / VERIFY).
         With max_attempts = 3 and k = 3 the rule fires only when all three attempts made no progress, and it then
         turns the REJECT into an ESCALATE; it saves calls only for k < max_attempts.
    fs1  one "can this be done?" call before the first repair attempt (AgentConfigExt.feasibility_check = True),
         prompt version a1-v4-fs1. {"feasible": false, "reason": ...} -> ESCALATE with that reason; a reply that is
         not a JSON object with a boolean "feasible" counts as feasible (repair proceeds) and is logged.

    res = run_repair_ext(task, code, before, verify_fn, client, AgentConfigExt(mode="full", stop_after_no_progress=3))
"""
import json
from dataclasses import dataclass

from igt.agents.intent_parser import _first_json_object
from igt.agents.repair_agent import AGENT_PROMPT_VERSION, AgentConfig, _Run
from igt.orchestrator.memory import Attempt, code_hash

FEASIBILITY_PROMPT_VERSION = "a1-v4-fs1"
NO_PROGRESS_ID = "np1"
DEFAULT_NO_PROGRESS_K = 3

FEASIBILITY_SYSTEM = """You check whether a data transformation request can be done correctly and safely with the given data, before anyone tries to fix the code.

You see the user's request, the input data schema, the read-only checks (if any) and the report of what went wrong with the first code.
Answer "feasible": false only when no code could satisfy the request, for example:
- it refers to columns or data that are not in the input (another file, a website, made-up values);
- it asks for two things that can't both hold;
- it is too ambiguous to know what result is wanted;
- it needs something unsafe (files, network, system commands).
If the first code simply has a bug, the request is feasible.

Reply with one JSON object and nothing else:
{"feasible": true, "reason": "<one sentence>"}
or
{"feasible": false, "reason": "<one sentence>"}"""


@dataclass
class AgentConfigExt(AgentConfig):
    stop_after_no_progress: int | None = None     # np1: k; None = the a1-v4 behaviour
    feasibility_check: bool = False               # fs1

    def __post_init__(self):
        super().__post_init__()
        k = self.stop_after_no_progress
        if k is not None and (not isinstance(k, int) or k < 1):
            raise ValueError("stop_after_no_progress must be a positive integer or None")


def ext_version(cfg):
    """Agent version id recorded in runs: a1-v4, a1-v4+np1-k3, a1-v4-fs1, a1-v4-fs1+np1-k2, ..."""
    v = FEASIBILITY_PROMPT_VERSION if getattr(cfg, "feasibility_check", False) else AGENT_PROMPT_VERSION
    k = getattr(cfg, "stop_after_no_progress", None)
    return v + (f"+{NO_PROGRESS_ID}-k{k}" if k else "")


def no_progress(prev, cur):
    """Same rule as eval/agent_metrics.no_progress (a test checks they agree)."""
    if cur[0] == "pass":
        return False
    if cur[0] in ("repeat", "exec"):
        return True
    if prev[0] in ("exec", "repeat"):
        return False
    return not (cur[1] < prev[1])


def parse_feasibility(text):
    """-> (feasible: bool, reason: str, malformed: bool). Malformed replies count as feasible."""
    raw = _first_json_object(text or "")
    try:
        obj = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("feasible"), bool):
        return True, "", True
    return obj["feasible"], str(obj.get("reason", ""))[:300], False


class _RunExt(_Run):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.np_k = getattr(self.cfg, "stop_after_no_progress", None)
        self.np_prev, self.np_count, self.np_stopped = None, 0, False
        self.feas_done = False

    # ---- np1 ----
    def _np_update(self, res):
        if self.np_prev is None:
            self.np_prev = res                    # the initial code: the reference point
            return
        self.np_count = self.np_count + 1 if no_progress(self.np_prev, res) else 0
        if res[0] != "repeat":
            self.np_prev = res

    def _np_should_stop(self):
        return self.np_k is not None and self.np_count >= self.np_k

    def _np_stop(self):
        if self.sm.state == "FAILED":
            self.sm.go("REPAIR")
        self.escalate_reason = f"no progress in {self.np_count} consecutive attempts (np1, k={self.np_k})"
        self.np_stopped = True
        self.ev("no_progress_stop", k=self.np_k, consecutive=self.np_count)
        self.sm.go("ESCALATED")

    def evaluate(self, code, attempt_no):
        super().evaluate(code, attempt_no)
        last = self.trace[-1]
        if last["event"] == "verified":
            res = ("pass", frozenset()) if last["status"] == "PASS" else ("fail", frozenset(last.get("failed") or []))
        else:
            res = ("exec", None)
        self._np_update(res)
        if attempt_no > 0 and self.sm.state == "FAILED" and self._np_should_stop():
            self._np_stop()

    def propose(self, code, attempt_tool_calls):
        if not self.mem.seen(code_hash(code)) or self.mem.repeat_count + 1 >= 2:
            return super().propose(code, attempt_tool_calls)     # new code, or the a1-v4 "repeated twice" escalation
        h = code_hash(code)                                       # first repeat: as a1-v4, plus the no-progress count
        self.n_fix += 1
        self.mem.repeat_count += 1
        self.ev("repeated_code", code_hash=h, times=self.mem.repeat_count)
        self.mem.add(Attempt(self.n_fix, code, h, "This exact code was already tried; it gives the same result.",
                             "repeated", attempt_tool_calls))
        self._np_update(("repeat", None))
        if self._np_should_stop():
            self._np_stop()
            return None
        self.sm.go("REJECTED") if self.n_fix >= self.L.max_attempts else None
        return "repeated"

    # ---- fs1 ----
    def feasibility_messages(self):
        text = self.report["coarse"] if self.cfg.report == "coarse" else self.report["text"]
        user = (f"{self.task_text()}\n\nWhat went wrong with the first code:\n{text}\n\n"
                'Can this request be done correctly and safely with this data? Reply with the JSON object only.')
        return [{"role": "system", "content": FEASIBILITY_SYSTEM}, {"role": "user", "content": user}]

    def repair(self):
        if (getattr(self.cfg, "feasibility_check", False) and not self.feas_done and self.sm.state == "FAILED"
                and self.cfg.mode != "none" and self.cfg.allow_escalate and self.n_fix < self.L.max_attempts):
            self.feas_done = True
            feasible, reason, malformed = parse_feasibility(self.llm(self.feasibility_messages()))
            self.ev("feasibility", feasible=feasible, reason=reason, malformed=malformed)
            if not feasible:
                self.sm.go("REPAIR")
                self.escalate_reason = f"infeasible: {reason}" if reason else "infeasible"
                self.sm.go("ESCALATED")
                return
        return super().repair()


def run_repair_ext(task, initial_code, before, verify_fn, client, config=None, contract_text=None, schema_text=None):
    """Same interface as repair_agent.run_repair; uses the extension run loop."""
    cfg = config or AgentConfigExt()
    if client.token_budget is None:
        client.token_budget = cfg.limits.token_budget
    return _RunExt(task, initial_code, before, verify_fn, client, cfg, contract_text, schema_text).run()
