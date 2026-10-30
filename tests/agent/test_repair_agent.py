"""Every agent path from README Phase 9 task 4, with the scripted backend (no GPU):
accept on first try, repair then accept, repeated code, attempt cap, tool error then recovery, unknown tool,
escalate, blocked code; plus the baselines (none / naive / feedback), limits, budget, faults and ablations."""
import json

import pandas as pd
import pytest

from igt.agents.repair_agent import AgentConfig, parse_action, run_repair
from igt.llm import LLMClient, ScriptedBackend
from igt.orchestrator import IllegalTransition, Limits, StateMachine
from igt.verifier import verify

B = pd.DataFrame({"host": ["a", "a", "b", "c"], "price": [10, 0, 30, 0]})
TASK = {"request": "Remove the listings with a price of 0."}
CONTRACT = {"must_change": [{"type": "rows_removed_where", "condition": {"column": "price", "op": "==", "value": 0}}],
            "must_preserve": [{"type": "rows_retained_where", "condition": {"column": "price", "op": "!=", "value": 0}},
                              {"type": "columns_unchanged", "target": "*", "except": []}]}
GOOD = "def transform(df):\n    return df[df['price'] != 0].copy()\n"
BAD = "def transform(df):\n    return df[df['price'] > 10].copy()\n"
BAD2 = "def transform(df):\n    return df[df['price'] > 20].copy()\n"
UNSAFE = "import os\ndef transform(df):\n    return df\n"
LIM = Limits(sandbox_timeout_s=20, sandbox_cpu_s=10)


def vf(after):
    return verify(CONTRACT, B, after)[0]


def tool(name, **args):
    return json.dumps({"action": "tool", "tool": name, "args": args})


def fix(code):
    return '{"action": "propose_fix"}\n```python\n' + code + "```"


def esc(reason="ambiguous"):
    return json.dumps({"action": "escalate", "reason": reason})


def run(initial, script, **cfg):
    be = ScriptedBackend(script)
    client = LLMClient(be)
    res = run_repair(TASK, initial, B, vf, client, AgentConfig(limits=cfg.pop("limits", LIM), **cfg),
                     contract_text=json.dumps(CONTRACT))
    return res, be


def test_accept_on_first_try_makes_no_llm_call():
    res, be = run(GOOD, [])
    assert res.outcome == "ACCEPT" and res.attempts == 0 and be.calls == []
    assert res.path == ["PREPARE", "GENERATE", "SAFETY_CHECK", "EXECUTE", "VERIFY", "ACCEPTED"]
    assert res.final_output is not None and len(res.final_output) == 2


def test_repair_with_tools_then_accept():
    res, be = run(BAD, [tool("diff_datasets"), tool("profile_column", column="price"), fix(GOOD)])
    assert res.outcome == "ACCEPT" and res.attempts == 1 and res.tool_calls == 2
    assert "rows_retained_where" in be.calls[0][1]["content"]          # failure report reached the agent
    assert "Observation" in be.calls[2][-1]["content"]                  # tool results fed back
    assert [e["tool"] for e in res.trace if e["event"] == "tool"] == ["diff_datasets", "profile_column"]


def test_diff_before_run_is_a_tool_error_then_recovery():
    res, _ = run(BAD, [tool("diff_datasets"), tool("run_transformation", code=BAD), tool("diff_datasets"), fix(GOOD)])
    errs = [e.get("error_type") for e in res.trace if e["event"] == "tool"]
    assert errs == [None, None, None] and res.outcome == "ACCEPT"       # the initial output is diffable right away
    res2, _ = run(BAD, [tool("profile_column", column="nope"), tool("profile_column", column="price"), fix(GOOD)])
    assert [e["error_type"] for e in res2.trace if e["event"] == "tool"] == ["tool_error", None] and res2.outcome == "ACCEPT"


def test_unknown_tool_is_observed_and_counted():
    res, _ = run(BAD, [tool("delete_everything"), fix(GOOD)])
    assert res.unknown_tool_calls == 1 and res.outcome == "ACCEPT"


def test_escalate():
    res, _ = run(BAD, [esc("request is ambiguous")])
    assert res.outcome == "ESCALATE" and res.escalate_reason == "request is ambiguous"


def test_blocked_fix_counts_as_unsafe_attempt_then_recovers():
    res, _ = run(BAD, [fix(UNSAFE), fix(GOOD)])
    assert res.unsafe_attempts == 1 and res.outcome == "ACCEPT" and res.attempts == 2
    assert any(e["event"] == "safety_block" for e in res.trace)


def test_blocked_code_via_run_transformation_tool_is_unsafe():
    res, _ = run(BAD, [tool("run_transformation", code=UNSAFE), fix(GOOD)])
    assert res.unsafe_attempts == 1 and res.outcome == "ACCEPT"


def test_attempt_cap_rejects():
    res, be = run(BAD, [fix(BAD2), fix(BAD2.replace("20", "25")), fix(BAD2.replace("20", "15"))],
                  limits=Limits(max_attempts=3, sandbox_timeout_s=20, sandbox_cpu_s=10))
    assert res.outcome == "REJECT" and res.attempts == 3 and len(be.calls) == 3


def test_repeated_code_observed_once_then_escalates():
    res, be = run(BAD, [fix(BAD), fix(BAD)], limits=Limits(max_attempts=5, sandbox_timeout_s=20, sandbox_cpu_s=10))
    assert res.outcome == "ESCALATE" and res.escalate_reason == "identical code proposed twice"
    assert "already tried" in be.calls[1][-1]["content"]


def test_repeated_code_ignores_formatting():
    res, _ = run(BAD, [fix(BAD.replace("df['price'] > 10", "df[ 'price' ]>10")), fix("# same code\n" + BAD)],
                 limits=Limits(max_attempts=5, sandbox_timeout_s=20, sandbox_cpu_s=10))
    assert res.outcome == "ESCALATE"


def test_tool_call_limit_forces_a_decision():
    script = [tool("inspect_dataset")] * 2 + [tool("inspect_dataset"), fix(GOOD)]
    res, be = run(BAD, script, limits=Limits(max_tool_calls_per_attempt=2, sandbox_timeout_s=20, sandbox_cpu_s=10))
    assert res.tool_calls == 2 and res.outcome == "ACCEPT"
    assert any(e["event"] == "tool_limit" for e in res.trace)


def test_malformed_action_reprompted_then_too_many_rejects():
    res, be = run(BAD, ["I think the filter is wrong.", fix(GOOD)])
    assert res.invalid_actions == 1 and res.outcome == "ACCEPT" and "no JSON action" in be.calls[1][-1]["content"]
    res2, _ = run(BAD, ["nope"] * 5, limits=Limits(max_invalid_actions=3, sandbox_timeout_s=20, sandbox_cpu_s=10))
    assert res2.outcome == "REJECT" and res2.invalid_actions == 4


def test_token_budget_rejects():
    res, _ = run(BAD, [tool("inspect_dataset")] * 20, limits=Limits(token_budget=3000, sandbox_timeout_s=20, sandbox_cpu_s=10))
    assert res.outcome == "REJECT" and any(e["event"] == "budget_exceeded" for e in res.trace)


def test_uncertain_verdict_escalates_without_repair():
    c = {"must_change": [{"type": "null_count", "target": "nope", "op": "eq", "value": 0}]}
    be = ScriptedBackend([])
    res = run_repair(TASK, BAD, B, lambda a: verify(c, B, a)[0], LLMClient(be), AgentConfig(limits=LIM))
    assert res.outcome == "ESCALATE" and be.calls == []


# ---------------- agent baselines ----------------
def test_mode_none_rejects_without_llm():
    res, be = run(BAD, [], mode="none")
    assert res.outcome == "REJECT" and be.calls == []


def test_naive_retry_gets_no_report():
    res, be = run(BAD, ["```python\n" + GOOD + "```"], mode="naive")
    assert res.outcome == "ACCEPT"
    u = be.calls[0][1]["content"]
    assert "failed verification" in u and "rows_retained_where" not in u and "price'] > 10" not in u


def test_feedback_retry_gets_report_and_code_but_no_tools():
    res, be = run(BAD, ["```python\n" + GOOD + "```"], mode="feedback")
    u = be.calls[0][1]["content"]
    assert res.outcome == "ACCEPT" and "rows_retained_where" in u and "df['price'] > 10" in u
    assert "run_transformation" not in be.calls[0][0]["content"]


# ---------------- ablations ----------------
def test_coarse_report_and_hidden_contract():
    _, be = run(BAD, [fix(GOOD)], report="coarse", show_contract=False)
    u = be.calls[0][1]["content"]
    assert "Verification failed." in u and "rows_retained_where" not in u


def test_no_memory_shows_only_last_attempt():
    _, be = run(BAD, [fix(BAD2), fix(GOOD)], memory=False)
    u = be.calls[1][1]["content"]
    assert "Attempt 1" in u and "Initial code" not in u


def test_no_escalate_ablation():
    _, be = run(BAD, [esc(), fix(GOOD)], allow_escalate=False)
    assert "escalate" not in be.calls[0][0]["content"].split("Rules:")[1].split("You can't")[0].lower()
    assert "not available" in be.calls[1][-1]["content"]


# ---------------- faults (T scenarios) ----------------
def test_injected_fault_then_retry_recovers():
    res, _ = run(BAD, [tool("profile_column", column="price"), tool("profile_column", column="price"), fix(GOOD)],
                 faults=[{"tool": "profile_column", "on_call": 1, "kind": "error"}])
    assert res.faults_fired == [dict(tool="profile_column", on_call=1, kind="error")] and res.outcome == "ACCEPT"
    assert [e["error_type"] for e in res.trace if e["event"] == "tool"] == ["injected_fault", None]


# ---------------- plumbing ----------------
def test_state_machine_blocks_illegal_jumps():
    sm = StateMachine()
    with pytest.raises(IllegalTransition):
        sm.go("EXECUTE")                       # can't skip GENERATE / SAFETY_CHECK
    sm.go("GENERATE"); sm.go("SAFETY_CHECK")
    with pytest.raises(IllegalTransition):
        sm.go("VERIFY")                        # can't skip EXECUTE


def test_parse_action_variants():
    assert parse_action(tool("inspect_dataset"))[0]["tool"] == "inspect_dataset"
    assert "transform" in parse_action(fix(GOOD))[0]["code"]
    assert parse_action('{"action": "propose_fix", "code": "def transform(df):\\n    return df"}')[0]["code"]
    assert parse_action('{"action": "propose_fix"}')[1].startswith("propose_fix needs the code")
    assert parse_action('{"action": "dance"}')[0] is None and parse_action("hello")[0] is None


def test_snapshot_never_modified_by_agent_code():
    before = B.copy()
    run(BAD, [fix("def transform(df):\n    df['price'] = 0\n    return df")],
        limits=Limits(max_attempts=1, sandbox_timeout_s=20, sandbox_cpu_s=10))
    pd.testing.assert_frame_equal(B, before)


# ---- a1-v2 parser fixes (dev run 1: 7B replies were read as invalid actions) ----
RENAME = "def transform(df):\n    return df.rename(columns={'a': 'b'})\n"


def test_dict_literal_in_code_is_not_the_json_action():
    a, err = parse_action('{"action": "propose_fix"}\n```python\n' + RENAME + "```")
    assert err is None and a["action"] == "propose_fix" and not a.get("implicit")
    a, err = parse_action("The rename used the wrong name.\n```python\n" + RENAME + "```")
    assert err is None and a["action"] == "propose_fix" and a["implicit"] and "rename" in a["code"]


def test_bare_code_block_is_an_implicit_fix_but_code_without_transform_is_not():
    a, _ = parse_action("```python\n" + GOOD + "```")
    assert a["implicit"] and "transform" in a["code"]
    assert parse_action("```python\ndef helper(df):\n    return df\n\ndef other(df):\n    return df\n```")[0] is None
    assert parse_action("```python\nx = 1\n```")[0] is None


def test_code_field_with_raw_newlines_and_fenced_json_tool():
    a, err = parse_action('{"action": "propose_fix", "code": "def transform(df):\n    return df"}')
    assert err is None and "return df" in a["code"]
    assert parse_action('```json\n{"action": "tool", "tool": "diff_datasets", "args": {}}\n```')[0]["tool"] == "diff_datasets"
    assert parse_action('```\n{"action": "tool", "tool": "diff_datasets", "args": {}}\n```')[0]["tool"] == "diff_datasets"


def test_implicit_fix_is_counted_and_replies_are_logged():
    res, be = run(BAD, ["```python\n" + GOOD + "```"])
    assert res.outcome == "ACCEPT" and res.implicit_fixes == 1 and res.invalid_actions == 0
    assert [e["reply"] for e in res.trace if e["event"] == "llm"][0].startswith("```python")
    assert be.calls[0][-1]["content"].rstrip().endswith("in the same reply.")


def test_blocked_initial_code_is_not_an_agent_unsafe_attempt():
    res, _ = run(UNSAFE, [fix(GOOD)])
    assert res.initial_blocked and res.unsafe_attempts == 0 and res.outcome == "ACCEPT"


def test_no_escalate_prompt_has_no_escalate_example():
    from igt.agents.repair_agent import FULL_SYSTEM, FULL_SYSTEM_NO_ESCALATE
    assert '"escalate"' in FULL_SYSTEM and '"escalate"' not in FULL_SYSTEM_NO_ESCALATE
