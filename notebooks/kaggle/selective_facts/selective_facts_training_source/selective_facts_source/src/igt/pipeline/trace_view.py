"""Readable view of one repair run (README Phase 10 "annotated example trajectories", Phase 12 demo / case studies).

    from igt.pipeline.trace_view import agent_steps, render_markdown
    steps = agent_steps(trace_record["trace"], initial_code)       # one entry per agent decision
    print(render_markdown(run_record, trace_record))

Input: one line of <system>/traces.jsonl (scenario_id or task_id, initial_code, trace) written by eval/runner_agent.py
or eval/runner_stage2.py, and optionally the matching line of runs.jsonl (outcome and the evaluator's label).

An agent step = one model reply and what followed it until the next reply: a tool call and its result, a proposed fix
and its check, an escalation, or an invalid action. Step 0 is the first check of the generator's code (not an agent
decision). These steps are the units labelled in the trajectory study (eval/trajectory_labels.py).
"""
from igt.agents.code_extract import extract_code
from igt.agents.repair_agent import parse_action

MAX_CODE_LINES = 25


def _check_text(e):
    if e["event"] == "verified":
        failed = ", ".join(e.get("failed") or [])
        return f"check: {e['status']}" + (f" (failed: {failed})" if failed else "")
    if e["event"] == "execute_failed":
        return f"run failed: {e.get('status')}: {str(e.get('error') or '')[:160]}"
    if e["event"] == "safety_block":
        return f"blocked by the safety checker: {e.get('reason')}"
    if e["event"] == "repeated_code":
        return f"same code as an earlier attempt (time {e.get('times')})"
    return None


def _code_of(reply):
    a, _ = parse_action(reply)
    if a and a.get("action") == "propose_fix":
        return a["code"]
    ex = extract_code(reply)
    return ex["code"] if ex["parse_ok"] else None


def agent_steps(trace, initial_code=None):
    """-> list of dict(n, kind, summary, code, outcome). kind: first_check | tool | fix | escalate | invalid | other."""
    steps, cur = [], None
    first = [e for e in trace[:3] if _check_text(e)]
    if first:
        steps.append(dict(n=0, kind="first_check", summary="the generator's code is checked", code=initial_code,
                          outcome=_check_text(first[0])))
    for e in trace:
        if e["event"] == "llm":
            a, err = parse_action(e.get("reply") or "")
            cur = dict(n=len(steps) if steps else 1, kind="other", summary="", code=None, outcome=None, reply=e.get("reply"))
            if a and a["action"] == "tool":
                args = {k: (f"<code, {len(str(v).splitlines())} lines>" if k == "code" else v) for k, v in a["args"].items()}
                cur.update(kind="tool", summary=f"calls {a['tool']}({', '.join(f'{k}={v}' for k, v in args.items())})",
                           code=a["args"].get("code"))
            elif a and a["action"] == "escalate":
                cur.update(kind="escalate", summary=f"escalates: {str(a.get('reason', ''))[:200]}")
            elif a and a["action"] == "propose_fix":
                cur.update(kind="fix", summary="proposes a fix" + (" (code only)" if a.get("implicit") else ""),
                           code=a["code"])
            else:
                code = _code_of(e.get("reply") or "")
                cur.update(kind="fix" if code else "invalid", summary="proposes a fix" if code else f"invalid reply: {err}",
                           code=code)
            steps.append(cur)
            continue
        if cur is None:
            continue
        if e["event"] == "tool":
            if cur["kind"] in ("invalid", "other"):        # stored reply cut at 4,000 chars: use the recorded call
                args = {k: v for k, v in (e.get("args") or {}).items()}
                cur.update(kind="tool", summary=f"calls {e['tool']}({', '.join(f'{k}={v}' for k, v in args.items())})")
            cur["outcome"] = "tool ok" if e.get("ok") else f"tool error: {e.get('error_type')}"
            if e.get("ok") and e.get("tool") == "run_transformation" and cur.get("code"):
                from igt.sandbox import check          # the trace doesn't store the tool's output: re-check the code
                g = check(cur["code"])
                if not g.ok:
                    cur["outcome"] = ("code did not parse: " if g.reason.startswith("syntax error") else
                                      "code blocked by the safety checker: ") + g.reason
        elif e["event"] == "propose_fix" and cur["kind"] in ("invalid", "other"):
            cur.update(kind="fix", summary="proposes a fix (reply too long to show)")
        elif e["event"] == "invalid_action":
            cur.update(kind="invalid", outcome=f"invalid action: {str(e.get('error'))[:160]}")
        elif _check_text(e):
            cur["outcome"] = _check_text(e)
    for i, s in enumerate(steps):
        s["n"] = i
    return steps


def _code_block(code):
    if not code:
        return []
    lines = code.rstrip().splitlines()
    more = len(lines) - MAX_CODE_LINES
    return ["```python", *lines[:MAX_CODE_LINES], *(["# ... %d more lines" % more] if more > 0 else []), "```"]


def render_markdown(run, trace, request=None, show_code=True):
    """run: runs.jsonl record (may be None); trace: traces.jsonl record."""
    rid = trace.get("scenario_id") or trace.get("task_id")
    out = [f"### {rid}" + (f" ({run.get('category') or run.get('preset')}, {run.get('system')})" if run else "")]
    if request:
        out += ["", f"**Request:** {request}"]
    if run:
        if run["outcome"] != "ACCEPT":
            label = "not accepted"
        else:
            label = {True: "correct", False: "wrong", None: "no reference output"}[run.get("final_correct")]
        out += ["", f"**Outcome:** {run['outcome']} · evaluator: {label} · attempts {run.get('attempts')} · "
                    f"tool calls {run.get('tool_calls')} · tokens {run.get('tokens')}"]
    for s in agent_steps(trace["trace"], trace.get("initial_code")):
        out += ["", f"**{s['n']}. {s['summary']}**" + (f" → {s['outcome']}" if s["outcome"] else "")]
        if show_code and s["code"] and s["kind"] in ("first_check", "fix"):
            out += _code_block(s["code"])
    return "\n".join(out) + "\n"
