"""Bounded, tool-using repair agent (README Phase 9, §4.6–4.8) and the agent baselines (Phase 10).

    res = run_repair(task, initial_code, before, verify_fn, client, config=AgentConfig(mode="full"))
    res.outcome  # ACCEPT / REJECT / ESCALATE ;  res.trace: one event per step ;  res.final_code / res.final_output

Modes (README Phase 10 agent baselines; same model, limits and budget for all):
    none      no repair: the Stage 1 outcome of the initial code
    naive     regenerate from request + schema, told only "verification failed"
    feedback  regenerate with the previous code and the full failure report, no tools
    full      failure report + tools + memory + escalation + the read-only contract, one action per step
Ablation switches (full mode): memory, report ("full" | "coarse"), show_contract, allow_escalate.

Full-mode action format (one per reply):
    {"action": "tool", "tool": "<name>", "args": {...}}
    {"action": "propose_fix"} followed by one ```python block defining transform(df)   (or "code" inside the JSON)
    {"action": "escalate", "reason": "..."}
Invariants: the before snapshot and the contract are read-only (the agent only proposes code); every code run goes
through the safety checker and sandbox; the verifier is deterministic; ground truth never enters the loop.
"""
import json
import re
from dataclasses import dataclass, field

from igt.agents.code_extract import extract_code
from igt.agents.intent_parser import _first_json_object
from igt.llm import BudgetExceeded
from igt.orchestrator import Limits, Memory, StateMachine
from igt.orchestrator.memory import Attempt, code_hash
from igt.pipeline.failure_report import failure_report
from igt.sandbox import check, safe_run
from igt.tools import FaultInjector, ToolCall, ToolContext, build_registry

MODES = ("none", "naive", "feedback", "full")

AGENT_PROMPT_VERSION = "a1-v3"   # a1-v1: dev run 1, almost no JSON action; a1-v2: dev run 2, R success 5/14,
                                 # 11 replies used the tool name as "action", 3 wrong accepts, 1 fabricated column

FULL_SYSTEM = """You repair pandas code. A function transform(df) was written for the user's request, but checking its output found problems. Find the cause and propose fixed code, or escalate to a human if the request can't be done safely or correctly.

Rules:
- One action per reply. Start the reply with the JSON object of the action. For propose_fix, put one ```python block after it. Nothing else.
- {"action": "tool", "tool": NAME, "args": {...}} calls a tool. Tools:
{tools}
- {"action": "propose_fix"} followed by a ```python block that defines transform(df) and returns a DataFrame. Only pandas (pd) and numpy (np); no files, network or printing.
- {"action": "escalate", "reason": "..."} hands the task to a human. Use it when the request is ambiguous, contradictory, refers to columns that don't exist, or needs something unsafe.
- You can't change the data or the checks, only the code.
- The result must match the request exactly: drop any helper column you add, keep each column's type unless the request asks to change it (e.g. use np.nan, not pd.NA, in float columns), and don't touch other columns or rows.
- Never invent data. If the request needs data that isn't in the input (another file, a website, made-up values), escalate.
- If the same check keeps failing because the request asks for two things that can't both hold, escalate instead of retrying.
- At most {max_tools} tool calls per attempt, then you must propose_fix or escalate. At most {max_attempts} fixes in total.

Example replies (one per turn):
{"action": "tool", "tool": "diff_datasets", "args": {"max_examples": 5}}

{"action": "propose_fix"}
```python
def transform(df):
    out = df.copy()
    return out[out["price"] > 0]
```

{"action": "escalate", "reason": "the column 'salary' does not exist in the data"}"""

NEXT_ACTION = ('Reply with your next action. For propose_fix, write {"action": "propose_fix"} and then the complete '
               "```python block defining transform(df), in the same reply.")

FULL_SYSTEM_NO_ESCALATE = FULL_SYSTEM.replace(
    '\n- {"action": "escalate", "reason": "..."} hands the task to a human. Use it when the request is ambiguous, '
    'contradictory, refers to columns that don\'t exist, or needs something unsafe.', "").replace(
    " Find the cause and propose fixed code, or escalate to a human if the request can't be done safely or correctly.",
    " Find the cause and propose fixed code.").replace("then you must propose_fix or escalate", "then you must propose_fix").replace(
    '\n\n{"action": "escalate", "reason": "the column \'salary\' does not exist in the data"}', "").replace(
    "If the request needs data that isn't in the input (another file, a website, made-up values), escalate.",
    "If the request needs data that isn't in the input (another file, a website, made-up values), leave it out.").replace(
    "\n- If the same check keeps failing because the request asks for two things that can't both hold, escalate instead of retrying.", "")

RETRY_SYSTEM = """You are an expert Python data engineer. Write pandas code that performs exactly the transformation the user asks for.

Rules:
- Define a function `transform(df)` that takes a pandas DataFrame and returns the transformed DataFrame.
- `df` is already loaded and is passed to the function. Do not read or write files and do not use the network.
- pandas (as pd) and numpy (as np) are available.
- Do not print anything.
- Reply with a single ```python code block containing the function, and nothing else."""


@dataclass
class AgentConfig:
    mode: str = "full"
    memory: bool = True
    report: str = "full"           # "full" | "coarse"
    show_contract: bool = True     # intent systems: the agent sees the (read-only) contract
    allow_escalate: bool = True
    limits: Limits = field(default_factory=Limits)
    faults: list = field(default_factory=list)   # FaultInjector rules (T scenarios)
    temperature: float = 0.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")


@dataclass
class RepairResult:
    outcome: str
    final_code: str | None
    final_output: object
    final_status: str
    path: list
    trace: list
    attempts: int
    tool_calls: int
    invalid_actions: int
    unsafe_attempts: int
    unknown_tool_calls: int
    escalate_reason: str | None
    tokens: int
    llm_calls: int
    faults_fired: list
    implicit_fixes: int = 0          # full mode: replies that were only a code block, read as propose_fix
    initial_blocked: bool = False    # the initial (generator) code was blocked by the safety checker
    normalized_actions: int = 0      # full mode: {"action": "<tool name>"} replies read as tool calls


_FENCE = re.compile(r"```([A-Za-z]*)[ \t]*\n(.*?)```", re.S)


def _strip_code_blocks(text):
    """Remove ```python blocks (and unlabeled blocks holding a def) so only prose / JSON is left."""
    return _FENCE.sub(lambda m: "" if m.group(1).lower() in ("python", "py") or
                      (not m.group(1) and "def " in m.group(2)) else m.group(0), text)


def _implicit_fix(text):
    """A reply that is only a code block defining transform(df) is read as propose_fix (marked implicit=True).
    The retry baselines already accept such replies, so the full agent gets the same leniency."""
    ex = extract_code(text)
    if ex["parse_ok"] and ex["func_name"] == "transform":
        return dict(action="propose_fix", code=ex["code"], implicit=True)
    return None


def parse_action(text):
    """-> (action dict, None) or (None, error message).
    The JSON action is searched outside ```python blocks, so dict literals in the code are not mistaken for it."""
    raw = _first_json_object(_strip_code_blocks(text))
    if not raw:
        a = _implicit_fix(text)
        if a:
            return a, None
        return None, 'no JSON action found; reply with one JSON object such as {"action": "tool", ...}'
    try:
        a = json.loads(raw, strict=False)        # strict=False: raw newlines inside a "code" string are allowed
    except json.JSONDecodeError as e:
        a = _implicit_fix(text)
        if a:
            return a, None
        return None, f"the action is not valid JSON ({e})"
    if isinstance(a, dict) and isinstance(a.get("action"), str) and a["action"] not in ("tool", "propose_fix", "escalate") \
            and isinstance(a.get("args"), dict) and "tool" not in a:
        # {"action": "<tool name>", "args": {...}}: the tool name was put in "action" (11 of 98 replies in dev run 2).
        # Read it as that tool call; an unknown name still reaches the registry and is counted as an unknown tool.
        a = dict(action="tool", tool=a["action"], args=a["args"], normalized=True)
    if not isinstance(a, dict) or a.get("action") not in ("tool", "propose_fix", "escalate"):
        imp = _implicit_fix(text)
        if imp:
            return imp, None
        return None, 'the JSON needs "action": "tool" | "propose_fix" | "escalate"'
    if a["action"] == "tool":
        if not isinstance(a.get("tool"), str):
            return None, 'a tool action needs "tool": "<name>"'
        if not isinstance(a.get("args", {}), dict):
            return None, '"args" must be a JSON object'
        a.setdefault("args", {})
    if a["action"] == "propose_fix":
        code = a.get("code")
        if not code:
            m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
            code = m.group(1) if m else None
        if not code:
            return None, ('propose_fix needs the code: write {"action": "propose_fix"} and then a ```python block '
                          "defining transform(df), in the same reply")
        a["code"] = code
    return a, None


class _Run:
    """One repair run. Kept small: the state machine decides what may happen next."""

    def __init__(self, task, initial_code, before, verify_fn, client, cfg, contract_text, schema_text):
        self.task, self.before, self.verify_fn, self.client, self.cfg = task, before, verify_fn, client, cfg
        self.contract_text, self.schema_text = contract_text, schema_text
        self.L = cfg.limits
        self.sm = StateMachine()
        self.mem = Memory(enabled=cfg.memory)
        self.trace = []
        self.ctx = ToolContext(before=before, verify_fn=verify_fn, sandbox_limits=self.L.sandbox())
        reg = build_registry(self.ctx)
        self.reg = FaultInjector(reg, cfg.faults) if cfg.faults else reg
        self.n_fix = self.tool_calls = self.invalid = self.unsafe = self.unknown = self.implicit = self.normalized = 0
        self.initial_blocked = False
        self.code, self.output, self.status, self.report = initial_code, None, None, None
        self.escalate_reason = None

    def ev(self, kind, **kw):
        self.trace.append(dict(step=len(self.trace), state=self.sm.state, event=kind, **kw))

    # ---- one evaluation of a piece of code: SAFETY_CHECK -> EXECUTE -> VERIFY ----
    def evaluate(self, code, attempt_no):
        h = code_hash(code)
        self.sm.go("SAFETY_CHECK")
        g = check(code)
        if not g.ok and g.reason.startswith("syntax error"):     # not a safety event: the code doesn't parse
            self.sm.go("FAILED")
            self.status, self.report = "parse_error", failure_report("execute", exec_status="parse_error",
                                                                     exec_error=g.reason, code_hash=h)
            self.ev("execute_failed", code_hash=h, status="parse_error", error=g.reason)
        elif not g.ok:
            if attempt_no == 0:
                self.initial_blocked = True      # the code generator's code, not an agent action
            else:
                self.unsafe += 1
            self.status, self.report = "blocked", failure_report("safety", exec_status="blocked", exec_error=g.reason, code_hash=h)
            self.ev("safety_block", code_hash=h, reason=g.reason)
            self.sm.go("FAILED")
        else:
            self.sm.go("EXECUTE")
            r = safe_run(code, self.before, "transform", **self.L.sandbox())
            if not r.ran_ok:
                self.status, self.report = r.status, failure_report("execute", exec_status=r.status, exec_error=r.error, code_hash=h)
                self.ev("execute_failed", code_hash=h, status=r.status, error=(r.error or "")[:300])
                self.sm.go("FAILED")
            else:
                self.sm.go("VERIFY")
                v = self.verify_fn(r.df_out)
                self.status, self.output = v.status, r.df_out
                self.ctx.last_output, self.ctx.last_code = r.df_out, code     # tools can diff it right away
                self.report = failure_report("verify", verdict=v, code_hash=h)
                self.ev("verified", code_hash=h, status=v.status, n_failures=len(v.failures),
                        failed=sorted({f.get("predicate_id") or f.get("check") or f.get("kind") for f in v.failures}))
                if v.status == "PASS":
                    self.sm.go("ACCEPTED")
                elif v.status == "UNCERTAIN":
                    self.sm.go("ESCALATED")
                    self.escalate_reason = "verifier uncertain"
                else:
                    self.sm.go("FAILED")
        self.code = code
        text = self.report["coarse"] if self.cfg.report == "coarse" else self.report["text"]
        self.mem.add(Attempt(attempt_no, code, h, text, self.status))

    def llm(self, messages):
        r = self.client.generate(messages, max_new_tokens=self.L.max_new_tokens, temperature=self.cfg.temperature,
                                 seed=len(self.trace))
        self.ev("llm", tokens=r.prompt_tokens + r.output_tokens, cached=r.cached, reply=r.text[:4000])
        return r.text

    # ---- prompts ----
    def task_text(self, with_contract=True):
        parts = [f"Request: {self.task['request']}"]
        if self.schema_text:
            parts.append(f"Input data:\n{self.schema_text}")
        if with_contract and self.contract_text and self.cfg.show_contract:
            parts.append(f"The checks (read-only intent contract):\n{self.contract_text}")
        return "\n\n".join(parts)

    def full_messages(self, observations):
        tools = "\n".join(f"  {s['name']}: {s['description']} args: {json.dumps(s['args']['properties'])}"
                          for s in self.reg.specs())
        tmpl = FULL_SYSTEM if self.cfg.allow_escalate else FULL_SYSTEM_NO_ESCALATE
        system = (tmpl.replace("{tools}", tools).replace("{max_tools}", str(self.L.max_tool_calls_per_attempt))
                  .replace("{max_attempts}", str(self.L.max_attempts)))
        user = f"{self.task_text()}\n\nHistory:\n{self.mem.history_text()}\n\n{NEXT_ACTION}"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for act_text, obs in observations:
            msgs += [{"role": "assistant", "content": act_text},
                     {"role": "user", "content": "Observation:\n" + json.dumps(obs, default=str)[:6000] + "\n\n" + NEXT_ACTION}]
        return msgs

    def retry_messages(self):
        # Retry baselines regenerate like the code generator (request + schema). Naive: told only that verification
        # failed. Feedback: + previous code + failure report. The contract itself is shown only to the full agent.
        user = f"{self.task_text(with_contract=False)}"
        if self.cfg.mode == "naive":
            user += "\n\nA previous attempt failed verification. Write the function again."
        else:
            last = self.mem.attempts[-1]
            text = self.report["coarse"] if self.cfg.report == "coarse" else self.report["text"]
            user += (f"\n\nYour previous code:\n```python\n{last.code}\n```\n\nVerification result:\n{text}\n\n"
                     "Write a corrected version of the function.")
        return [{"role": "system", "content": RETRY_SYSTEM}, {"role": "user", "content": user}]

    # ---- repair loop ----
    def propose(self, code, attempt_tool_calls):
        h = code_hash(code)
        self.n_fix += 1
        if self.mem.seen(h):
            self.mem.repeat_count += 1
            self.ev("repeated_code", code_hash=h, times=self.mem.repeat_count)
            if self.mem.repeat_count >= 2:
                self.sm.go("ESCALATED")
                self.escalate_reason = "identical code proposed twice"
                return
            self.mem.add(Attempt(self.n_fix, code, h, "This exact code was already tried; it gives the same result.",
                                 "repeated", attempt_tool_calls))
            self.sm.go("REJECTED") if self.n_fix >= self.L.max_attempts else None
            return "repeated"
        self.evaluate(code, self.n_fix)
        self.mem.attempts[-1].tool_calls = attempt_tool_calls

    def repair(self):
        while self.sm.state in ("FAILED", "REPAIR"):
            if self.sm.state == "FAILED":
                if self.cfg.mode == "none" or self.n_fix >= self.L.max_attempts:
                    self.sm.go("REJECTED")
                    return
                self.sm.go("REPAIR")
            if self.cfg.mode in ("naive", "feedback"):
                ex = extract_code(self.llm(self.retry_messages()))
                if not ex["parse_ok"]:
                    self.invalid += 1
                    self.n_fix += 1
                    self.ev("invalid_action", error=ex.get("parse_reason"))
                    self.mem.add(Attempt(self.n_fix, "", "", "The reply contained no usable function.", "parse_error"))
                    if self.n_fix >= self.L.max_attempts:
                        self.sm.go("REJECTED")
                    else:
                        self.sm.go("REPAIR")
                    continue
                self.ev("propose_fix", code_hash=code_hash(ex["code"]))
                if self.propose(ex["code"], []) == "repeated" and self.sm.state == "REPAIR":
                    self.sm.go("REPAIR")
                continue
            self.full_attempt()

    def full_attempt(self):
        observations, calls, invalid_here = [], [], 0
        while True:
            text = self.llm(self.full_messages(observations))
            a, err = parse_action(text)
            if a is None or (a["action"] == "escalate" and not self.cfg.allow_escalate):
                err = err or "escalate is not available; propose_fix instead"
                self.invalid += 1
                invalid_here += 1
                self.ev("invalid_action", error=err)
                if invalid_here > self.L.max_invalid_actions:
                    self.sm.go("REJECTED")
                    return
                observations.append((text, dict(error=err)))
                self.sm.go("REPAIR")
                continue
            if a["action"] == "escalate":
                self.escalate_reason = str(a.get("reason", ""))[:300]
                self.ev("escalate", reason=self.escalate_reason)
                self.sm.go("ESCALATED")
                return
            if a["action"] == "tool":
                if len(calls) >= self.L.max_tool_calls_per_attempt:
                    self.invalid += 1
                    invalid_here += 1
                    self.ev("tool_limit", tool=a["tool"])
                    if invalid_here > self.L.max_invalid_actions:
                        self.sm.go("REJECTED")
                        return
                    observations.append((text, dict(error="tool-call limit for this attempt reached; propose_fix or escalate")))
                    self.sm.go("REPAIR")
                    continue
                self.normalized += bool(a.get("normalized"))
                res = self.reg.call(ToolCall(a["tool"], a["args"]))
                self.tool_calls += 1
                if (res.error or {}).get("type") == "unknown_tool":
                    self.unknown += 1
                if a["tool"] == "run_transformation" and res.ok and res.output.get("status") == "blocked":
                    self.unsafe += 1
                calls.append(dict(tool=a["tool"], args=a["args"], ok=res.ok, error_type=(res.error or {}).get("type")))
                self.ev("tool", tool=a["tool"], args={k: (v if k != "code" else code_hash(v)) for k, v in a["args"].items()},
                        ok=res.ok, error_type=(res.error or {}).get("type"))
                observations.append((text, res.to_observation()))
                self.sm.go("REPAIR")
                continue
            self.ev("propose_fix", code_hash=code_hash(a["code"]), implicit=bool(a.get("implicit")))
            self.implicit += bool(a.get("implicit"))
            out = self.propose(a["code"], calls)
            if out == "repeated" and self.sm.state == "REPAIR":
                observations.append((text, dict(error="this exact code was already tried and failed; change it")))
                self.sm.go("REPAIR")
                calls = []
                continue
            return

    def run(self):
        self.sm.go("GENERATE")
        self.ev("initial_code", code_hash=code_hash(self.code))
        try:
            self.evaluate(self.code, 0)
            self.repair()
        except BudgetExceeded:
            self.ev("budget_exceeded")
            if self.sm.state in ("FAILED", "REPAIR"):
                self.sm.go("REJECTED")
        return RepairResult(
            outcome=self.sm.outcome, final_code=self.code, final_output=self.output if self.sm.outcome == "ACCEPT" else None,
            final_status=self.status, path=self.sm.path, trace=self.trace, attempts=self.n_fix, tool_calls=self.tool_calls,
            invalid_actions=self.invalid, unsafe_attempts=self.unsafe,
            unknown_tool_calls=self.unknown, escalate_reason=self.escalate_reason,
            tokens=self.client.tokens_used, llm_calls=self.client.n_calls,
            faults_fired=getattr(self.reg, "fired", []), implicit_fixes=self.implicit,
            initial_blocked=self.initial_blocked, normalized_actions=self.normalized)


def run_repair(task, initial_code, before, verify_fn, client, config=None, contract_text=None, schema_text=None):
    """task: dict with at least 'request'. verify_fn(after DataFrame) -> Verdict (intent or conventional)."""
    cfg = config or AgentConfig()
    if client.token_budget is None:
        client.token_budget = cfg.limits.token_budget
    return _Run(task, initial_code, before, verify_fn, client, cfg, contract_text, schema_text).run()
