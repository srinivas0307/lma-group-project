"""Extensions #11 (np1, no-progress stop) and #12 (fs1, feasibility step): scripted runs, no GPU."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.agent_metrics import no_progress as metric_no_progress
from eval.ext_replay_no_progress import replay
from igt.agents.repair_agent import AGENT_PROMPT_VERSION, AgentConfig, run_repair
from igt.agents.repair_agent_ext import (AgentConfigExt, ext_version, no_progress, parse_feasibility,
                                         run_repair_ext)
from igt.llm import LLMClient, ScriptedBackend
from tests.agent.test_repair_agent import B, BAD, BAD2, CONTRACT, GOOD, LIM, TASK, fix, tool, vf

ROOT = Path(__file__).resolve().parents[2]
BAD3 = "def transform(df):\n    return df[df['price'] > 25].copy()\n"
CRASH = "def transform(df):\n    return df['nope']\n"


def run_ext(initial, script, **cfg):
    be = ScriptedBackend(script)
    res = run_repair_ext(TASK, initial, B, vf, LLMClient(be),
                         AgentConfigExt(limits=cfg.pop("limits", LIM), **cfg), contract_text=json.dumps(CONTRACT))
    return res, be


def run_base(initial, script):
    return run_repair(TASK, initial, B, vf, LLMClient(ScriptedBackend(script)), AgentConfig(limits=LIM),
                      contract_text=json.dumps(CONTRACT))


def test_no_progress_rule_matches_the_metric():
    F = frozenset
    cases = [(("fail", F("ab")), ("fail", F("a"))), (("fail", F("a")), ("fail", F("a"))), (("exec", None), ("fail", F("a"))),
             (("fail", F("a")), ("exec", None)), (("fail", F("a")), ("repeat", None)), (("fail", F("a")), ("pass", F()))]
    for prev, cur in cases:
        assert no_progress(prev, cur) == metric_no_progress(prev, cur)


@pytest.mark.parametrize("script", [[fix(GOOD)], [fix(BAD2), fix(GOOD)], [fix(BAD2), fix(BAD3), fix(CRASH)]])
def test_off_by_default_is_the_a1v4_agent(script):
    a, _ = run_ext(BAD, list(script))
    b = run_base(BAD, list(script))
    assert (a.outcome, a.path, a.attempts, a.escalate_reason) == (b.outcome, b.path, b.attempts, b.escalate_reason)
    assert ext_version(AgentConfigExt()) == AGENT_PROMPT_VERSION


def test_zero_no_progress_attempts_never_stop():
    res, _ = run_ext(BAD, [fix(GOOD)], stop_after_no_progress=1)
    assert res.outcome == "ACCEPT"


def test_one_no_progress_attempt():
    script = [fix(BAD2), fix(GOOD)]
    res1, be1 = run_ext(BAD, list(script), stop_after_no_progress=1)
    assert res1.outcome == "ESCALATE" and "no progress" in res1.escalate_reason and res1.attempts == 1
    assert res1.path[-2:] == ["REPAIR", "ESCALATED"] and len(be1.calls) == 1        # the second call is saved
    res2, _ = run_ext(BAD, list(script), stop_after_no_progress=2)
    assert res2.outcome == "ACCEPT"


def test_two_and_three_no_progress_attempts():
    script = [fix(BAD2), fix(BAD3), fix(CRASH)]
    res2, be2 = run_ext(BAD, list(script), stop_after_no_progress=2)
    assert res2.outcome == "ESCALATE" and res2.attempts == 2 and len(be2.calls) == 2
    res3, be3 = run_ext(BAD, list(script), stop_after_no_progress=3)       # k = max_attempts: only relabels REJECT
    base = run_base(BAD, list(script))
    assert base.outcome == "REJECT" and res3.outcome == "ESCALATE" and len(be3.calls) == 3 == base.llm_calls


def test_repeated_code_counts_as_no_progress():
    res, _ = run_ext(BAD, [fix(BAD2), fix(BAD2)], stop_after_no_progress=2)
    assert res.outcome == "ESCALATE" and "no progress" in res.escalate_reason


def test_offline_replay_equals_the_real_run():
    scripts = [[fix(GOOD)], [fix(BAD2), fix(GOOD)], [fix(BAD2), fix(BAD3), fix(CRASH)], [tool("diff_datasets"), fix(BAD2), fix(GOOD)]]
    for k in (1, 2, 3):
        for i, s in enumerate(scripts):
            base = run_base(BAD, list(s))
            rec = dict(scenario_id=f"s{i}", seed=0, category="R", outcome=base.outcome, final_correct=None,
                       tokens=base.tokens, llm_calls=base.llm_calls)
            rep = replay([rec], [dict(scenario_id=f"s{i}", seed=0, trace=base.trace)], k)[0]
            real, _ = run_ext(BAD, list(s), stop_after_no_progress=k)
            assert rep["outcome"] == real.outcome, (k, i)
            assert rep["llm_calls"] == real.llm_calls, (k, i)


def test_invalid_k_is_refused():
    with pytest.raises(ValueError):
        AgentConfigExt(stop_after_no_progress=0)


# ---------------- fs1 ----------------
def test_parse_feasibility():
    assert parse_feasibility('{"feasible": false, "reason": "no column salary"}') == (False, "no column salary", False)
    assert parse_feasibility('{"feasible": true}')[0] is True
    assert parse_feasibility("I think it is fine") == (True, "", True)
    assert parse_feasibility('{"feasible": "no"}') == (True, "", True)


def test_infeasible_escalates_before_repair():
    res, be = run_ext(BAD, ['{"feasible": false, "reason": "contradictory"}', fix(GOOD)], feasibility_check=True)
    assert res.outcome == "ESCALATE" and res.escalate_reason == "infeasible: contradictory" and res.attempts == 0
    assert len(be.calls) == 1 and res.path[-2:] == ["REPAIR", "ESCALATED"]
    assert ext_version(AgentConfigExt(feasibility_check=True)) == "a1-v4-fs1"


def test_feasible_and_malformed_replies_repair_as_usual():
    for first in ('{"feasible": true, "reason": "a bug"}', "not json at all"):
        res, be = run_ext(BAD, [first, fix(GOOD)], feasibility_check=True)
        assert res.outcome == "ACCEPT" and len(be.calls) == 2
        ev = [e for e in res.trace if e["event"] == "feasibility"][0]
        assert ev["malformed"] == (first == "not json at all")


def test_no_feasibility_call_when_the_first_code_passes():
    res, be = run_ext(GOOD, [], feasibility_check=True)
    assert res.outcome == "ACCEPT" and be.calls == []


def test_versions_combine():
    assert ext_version(AgentConfigExt(stop_after_no_progress=3)) == "a1-v4+np1-k3"
    assert ext_version(AgentConfigExt(stop_after_no_progress=2, feasibility_check=True)) == "a1-v4-fs1+np1-k2"


# ---------------- runner wrappers (subprocess, so the module patch can't leak into other tests) ----------------
def test_agent_wrapper_names_and_records(tmp_path):
    cmd = [sys.executable, "eval/runner_agent_ext.py", "--system", "smoke", "--mode", "full", "--backend",
           "reference_policy", "--limit", "2", "--out", str(tmp_path)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    d = tmp_path / "smoke_np1_k3"
    runs = [json.loads(l) for l in (d / "runs.jsonl").read_text().splitlines()]
    assert {r["agent_prompt_version"] for r in runs} == {"a1-v4+np1-k3"} and len(runs) == 2
    assert json.loads((d / "run_info_ext.json").read_text())["stop_after_no_progress"] == 3
    p = subprocess.run(cmd + ["--stop-after-no-progress", "1"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and (tmp_path / "smoke_np1_k1").exists()        # other k -> other folder


def test_stage2_wrapper(tmp_path):
    cmd = [sys.executable, "eval/runner_stage2_ext.py", "--preset", "gold_contract", "--backend", "pipeline_check",
           "--limit", "1", "--out", str(tmp_path), "--feasibility-check", "--stop-after-no-progress", "2"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    info = json.loads((tmp_path / "gold_contract_np1_k2_fs1" / "run_info.json").read_text())
    assert info["config"]["agent_prompt_version"] == "a1-v4-fs1+np1-k2"
