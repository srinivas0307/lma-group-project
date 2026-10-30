"""Agent scenarios, runner and metrics (README Phase 10 task 1: check metrics on trajectories with known answers)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.agent_metrics import paired, summarize

ROOT = Path(__file__).resolve().parents[2]
SCEN_POOLS = [(ROOT / "archive/agent_scenarios/rct_v1.jsonl", ROOT / "archive/pools/pool_v2/pool_v2.jsonl"),
              (ROOT / "archive/agent_scenarios/rct_v2.jsonl", ROOT / "archive/pools/pool_v3/pool_v3.jsonl"),
              (ROOT / "archive/agent_scenarios/rct_v3.jsonl", ROOT / "benchmark/pool/pool_v4.jsonl"),
              (ROOT / "benchmark/agent_scenarios/rct_v4.jsonl", ROOT / "benchmark/pool/pool_v4.jsonl")]


def run(r_cat, outcome, correct, **kw):
    base = dict(scenario_id=kw.pop("sid", "x"), seed=0, category=r_cat, outcome=outcome, final_correct=correct,
                attempts=1, tool_calls=0, llm_calls=0, tokens=0, latency_s=0.0, invalid_actions=0, unsafe_attempts=0,
                unknown_tool_calls=0, faults_fired=[], new_failures_after_repair=[], tool_trace=[])
    base.update(kw)
    return base


def test_metrics_on_known_trajectories():
    runs = [run("R", "ACCEPT", True, sid="r1"), run("R", "ACCEPT", False, sid="r2"),
            run("R", "ESCALATE", None, sid="r3", new_failures_after_repair=["p1"]), run("R", "REJECT", None, sid="r4"),
            run("C", "ACCEPT", True, sid="c1"), run("C", "ACCEPT", True, sid="c2", llm_calls=2),
            run("T", "ACCEPT", True, sid="t1", faults_fired=[{"tool": "x"}]), run("T", "REJECT", None, sid="t2"),
            run("R", "REJECT", None, sid="r5", unsafe_attempts=1,
                tool_trace=[dict(tool="a", ok=False, error_type="unknown_tool"), dict(tool="b", ok=True, error_type=None)])]
    m = summarize(runs)
    assert m["R"]["n"] == 5 and m["R"]["repair_success"] == 0.2 and m["R"]["escalate"] == 0.2 and m["R"]["regression"] == 0.2
    assert m["C"]["correct_accept"] == 1.0 and m["C"]["unnecessary_action"] == 0.5
    assert m["T"]["fault_fired"] == 1 and m["T"]["recovery"] == 1.0 and m["T"]["success"] == 0.5
    assert m["verifier_gaming"] == 0.2                 # 1 wrong of 5 accepts
    assert m["tools"]["valid_call_rate"] == 0.5 and m["safety"]["runs_with_unsafe_attempt"] == 1
    p = paired({"a": [run("R", "ACCEPT", True, sid="r1"), run("R", "REJECT", None, sid="r2")],
                "b": [run("R", "REJECT", None, sid="r1"), run("R", "REJECT", None, sid="r2")]})
    assert p["a vs b"]["only_first"] == 1 and p["a vs b"]["only_second"] == 0


@pytest.mark.parametrize("scen_path, pool_path", SCEN_POOLS, ids=["rct_v1", "rct_v2", "rct_v3", "rct_v4"])
def test_scenarios_file_is_well_formed_and_leak_free(scen_path, pool_path):
    if not scen_path.exists() or not pool_path.exists():
        pytest.skip("scenarios or pool not built")
    S = [json.loads(l) for l in scen_path.read_text().splitlines()]
    pool = {json.loads(l)["candidate_id"]: json.loads(l) for l in pool_path.read_text().splitlines()}
    assert len({s["id"] for s in S}) == len(S)
    for s in S:
        c = pool[s["candidate_id"]]
        assert s["initial_code"] == (c["code"] or "") and s["task_id"] == c["task_id"]
        assert not ({"is_correct", "reference_code", "reason"} & set(s)), "ground truth must not be in scenarios"
        if s["category"] == "C":
            assert c["is_correct"]
        else:
            assert not c["is_correct"]
        assert bool(s["faults"]) == (s["category"] == "T")


def test_runner_end_to_end_with_reference_policy(tmp_path):
    cmd = [sys.executable, str(ROOT / "eval/runner_agent.py"), "--system", "smoke", "--mode", "full",
           "--backend", "reference_policy", "--splits", "dev", "--limit", "4", "--out", str(tmp_path)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    runs = [json.loads(l) for l in (tmp_path / "smoke" / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 4 and all(r["category"] == "R" for r in runs)
    assert all(r["outcome"] in ("ACCEPT", "ESCALATE") for r in runs)
    assert all(r["final_correct"] for r in runs if r["outcome"] == "ACCEPT")
    info = json.loads((tmp_path / "smoke" / "run_info.json").read_text())
    assert info["valid_result"] is False                  # the reference policy is never a result
    p2 = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)        # resumable: nothing re-run
    assert len((tmp_path / "smoke" / "runs.jsonl").read_text().splitlines()) == 4 and "1/4" not in p2.stdout


def test_escalation_and_safety_metrics():
    runs = [run("E", "ESCALATE", None, sid="e1"), run("E", "REJECT", None, sid="e2"), run("R", "ESCALATE", None, sid="r1"),
            run("R", "ACCEPT", True, sid="r2"), run("S", "REJECT", None, sid="s1"), run("S", "ACCEPT", None, sid="s2", unsafe_attempts=1)]
    m = summarize(runs)
    assert m["E"]["escalation_recall"] == 0.5 and m["E"]["escalation_precision"] == 0.5 and m["E"]["escalation_f1"] == 0.5
    assert m["S"]["safe_outcome"] == 0.5 and m["S"]["accepted_safe_part"] == 0.5 and m["S"]["unsafe_attempt_runs"] == 1


def test_es_scenarios_are_well_formed():
    """E contracts: valid shape and no do-nothing pass. S contracts: the safe part is satisfiable by safe code."""
    import pandas as pd
    import yaml
    from igt.contracts import load_contract
    from igt.verifier import verify
    es = [yaml.safe_load(f.read_text()) for d in ("E", "S") for f in sorted((ROOT / "benchmark/agent_scenarios" / d).glob("*.yaml"))]
    assert len([s for s in es if s["category"] == "E"]) >= 12 and len([s for s in es if s["category"] == "S"]) >= 10
    assert len({s["id"] for s in es}) == len(es)
    for s in es:
        load_contract(s["contract"])
        before = pd.read_csv(ROOT / s["data_file"])
        v, _ = verify(s["contract"], before, before.copy())
        assert v.status != "PASS", (s["id"], "a do-nothing output passes")
        assert s["expected_outcome"] == ("ESCALATE" if s["category"] == "E" else "REJECT_OR_ESCALATE")


def test_runner_handles_es_with_generated_initial_code(tmp_path):
    cmd = [sys.executable, str(ROOT / "eval/runner_agent.py"), "--system", "smoke_es", "--mode", "full",
           "--backend", "reference_policy", "--splits", "dev", "--categories", "E", "S", "--out", str(tmp_path)]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    runs = [json.loads(l) for l in (tmp_path / "smoke_es" / "runs.jsonl").read_text().splitlines()]
    assert {r["category"] for r in runs} == {"E", "S"} and all(r["initial_generated"] for r in runs)
    assert all(r["outcome"] in ("ESCALATE", "REJECT") for r in runs) and all(r["final_correct"] is None for r in runs)
