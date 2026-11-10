"""Demo command (eval/demo.py) and the trace viewer (igt.pipeline.trace_view), without any model."""
import json
from pathlib import Path

import pytest
import yaml

from eval import demo, show_trace
from igt.pipeline.trace_view import agent_steps, render_markdown

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "results" / "stage2_agent" / "agent_full"
TASK = yaml.safe_load((ROOT / "benchmark/tasks/titanic/titanic_filt_003.yaml").read_text())
GOOD = 'def transform(df):\n    return df[~(df["Age"] < 18)].copy()\n'
BAD = 'def transform(df):\n    return df[df["Age"] >= 18].copy()\n'          # also drops unknown ages


def test_task_mode_shows_intent_catching_what_conventional_misses(capsys):
    r = demo.main(["task", "titanic_conv_002", "--candidate", "missed"])
    assert r["correct"] is False and r["conventional"] == "ACCEPT" and r["intent"] == "REJECT"
    assert "Intent verifier" in capsys.readouterr().out


def test_live_mode_scripted_parse_generate_repair(tmp_path, capsys):
    (tmp_path / "c.json").write_text(json.dumps(TASK["gold_contract"]))
    (tmp_path / "code.py").write_text(BAD)
    (tmp_path / "agent.json").write_text(json.dumps(['{"action": "propose_fix"}\n```python\n' + GOOD + "```"]))
    r = demo.main(["live", "--backend", "scripted", "--data", str(ROOT / TASK["data_file"]), "--request", TASK["request"],
                   "--scripted-contract", str(tmp_path / "c.json"), "--scripted-code", str(tmp_path / "code.py"),
                   "--scripted-agent", str(tmp_path / "agent.json"), "--save-output", str(tmp_path / "out.csv")])
    out = capsys.readouterr().out
    assert r["outcome"] == "ACCEPT" and r["attempts"] == 1
    assert "check: VIOLATION" in out and "check: PASS" in out and (tmp_path / "out.csv").exists()


def test_live_mode_invalid_contract_escalates(tmp_path):
    (tmp_path / "c.json").write_text('{"contract_version": "1.0", "must_change": [{"type": "no_such"}]}')
    r = demo.main(["live", "--backend", "scripted", "--data", str(ROOT / TASK["data_file"]), "--request", "x",
                   "--scripted-contract", str(tmp_path / "c.json")])
    assert r == dict(outcome="ESCALATE", stage="parse")


def test_agent_steps_from_a_synthetic_trace():
    trace = [dict(step=0, event="initial_code", code_hash="a"),
             dict(step=1, event="verified", code_hash="a", status="VIOLATION", failed=["c1"]),
             dict(step=2, event="llm", reply='{"action": "tool", "tool": "diff_datasets", "args": {}}'),
             dict(step=3, event="tool", tool="diff_datasets", ok=True, error_type=None),
             dict(step=4, event="llm", reply='{"action": "propose_fix"}\n```python\n' + GOOD + "```"),
             dict(step=5, event="propose_fix", code_hash="b"),
             dict(step=6, event="verified", code_hash="b", status="PASS", failed=[])]
    s = agent_steps(trace, BAD)
    assert [x["kind"] for x in s] == ["first_check", "tool", "fix"]
    assert s[0]["outcome"].startswith("check: VIOLATION") and s[2]["outcome"] == "check: PASS"
    md = render_markdown(dict(outcome="ACCEPT", final_correct=True, attempts=1, tool_calls=1, tokens=10, category="R",
                              system="x"), dict(scenario_id="R_x", initial_code=BAD, trace=trace))
    assert "calls diff_datasets" in md and "evaluator: correct" in md


@pytest.mark.skipif(not (AGENT / "traces.jsonl").exists(), reason="agent test run not downloaded")
def test_replay_and_show_trace_on_the_recorded_run():
    assert demo.main(["replay", str(AGENT), "R_test_003"])["outcome"] == "ACCEPT"
    assert "### R_test_003" in show_trace.main([str(AGENT), "R_test_003", "--no-code"])


@pytest.mark.skipif(not (AGENT / "traces.jsonl").exists(), reason="agent test run not downloaded")
def test_trajectory_sheets_sample_and_score(tmp_path):
    from eval import trajectory_labels as tl
    m = tl.main(["sample", "--out", str(tmp_path / "t")])
    assert len(m["runs"]) == 30 and len(m["second_labeller"]) == 10
    files = sorted((tmp_path / "t").glob("*.yaml"))
    for i, f in enumerate(files[:4]):                       # label 4 sheets; the first one twice, with one disagreement
        s = yaml.safe_load(f.read_text())
        for j, st in enumerate(s["steps"]):
            st["label"] = "useful" if j % 2 == 0 else "redundant"
            if "label_b" in st:
                st["label_b"] = st["label"] if j else "wrong"
        s.update(status="done", labeller="test")
        f.write_text(yaml.dump(s, sort_keys=False))
    r = tl.main(["score", "--sheets", str(tmp_path / "t"), "--out-dir", str(tmp_path)])
    assert r["runs"] == 4 and r["steps"] > 0 and r["overall"]["wrong"] == 0.0
    assert (tmp_path / "trajectories_summary.md").exists()
    assert tl.main(["score", "--sheets", str(tmp_path / "t"), "--check"])["done"] == 4


def test_kappa():
    from eval.trajectory_labels import kappa
    assert kappa([("useful", "useful"), ("wrong", "wrong")]) == 1.0
    assert kappa([("useful", "wrong"), ("wrong", "useful")]) < 0
