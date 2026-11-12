"""Prompt isolation, submission protocol and real subprocess environment checks."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from eval import runner_stage2
from eval.runner_stage2_prompts import main
from igt.agents import code_generator, code_generator_v2, repair_agent, repair_agent_v5
from igt.llm import LLMClient, ScriptedBackend
from igt.sandbox import executor
from igt.sandbox.environment import sandbox_environment
from igt.types import Verdict, check_failure

ROOT = Path(__file__).resolve().parents[1]


def test_generator_variants_leave_original_and_no_contract_messages_unchanged():
    schema = code_generator.load_prompt_schema(ROOT)
    task = {"data_file": next(iter(schema)), "request": "Change only the requested values."}
    contract = {"must_change": [], "must_preserve": []}
    original = code_generator.build_generator_messages(task, schema, contract)
    revised = code_generator_v2.build_generator_messages(task, schema, contract)
    assert original[1] == revised[1]
    assert revised[0]["content"] != original[0]["content"]
    assert code_generator_v2.build_generator_messages(task, schema) == code_generator.build_messages(task, schema)
    assert code_generator.PROMPT_HASH == "ebbe8bceae71"
    assert code_generator.CONTRACT_PROMPT_HASH == "243a463f9777"


def test_tested_success_requires_submission_and_preserves_original_prompt():
    before = pd.DataFrame({"amount": [1, 2]})
    good = 'def transform(df):\n    return df.assign(amount=df["amount"] + 1)\n'
    bad = 'def transform(df):\n    return df.copy()\n'

    def check(after):
        return Verdict("PASS") if after["amount"].tolist() == [2, 3] else Verdict.combine([
            check_failure("amount", after["amount"].tolist(), [2, 3])])

    original_system = repair_agent.FULL_SYSTEM

    def submit(messages):
        assert "Neither tool submits it" in messages[0]["content"]
        assert '"status": "PASS"' in messages[-1]["content"]
        return '{"action":"propose_fix"}\n```python\n' + good + '```'

    backend = ScriptedBackend([
        json.dumps({"action": "tool", "tool": "run_transformation", "args": {"code": good}}),
        '{"action":"tool","tool":"run_verifier","args":{}}', submit,
    ])
    result = repair_agent_v5.run_repair({"request": "Increase amount by one."}, bad, before, check,
                                      LLMClient(backend))
    assert result.outcome == "ACCEPT" and result.attempts == 1 and result.tool_calls == 2
    assert result.final_code == good
    assert repair_agent.FULL_SYSTEM == original_system and repair_agent.AGENT_PROMPT_VERSION == "a1-v4"


def test_sandbox_probe_really_uses_explicit_package_root_and_restores_runner():
    original = executor._RUNNER
    packages = Path(pd.__file__).resolve().parent.parent
    with sandbox_environment(packages) as info:
        assert info["pandas"] == "2.3.3" and info["numpy"] == "2.1.3"
        assert Path(info["pandas_file"]).is_relative_to(packages)
        assert info["executable"] == sys.executable
        assert executor._RUNNER != original
    assert executor._RUNNER == original


def test_probe_rejects_mismatched_child_environment(monkeypatch):
    original = executor._RUNNER
    class Result:
        ran_ok = True
        df_out = pd.DataFrame({"pandas": ["wrong"], "numpy": ["2.1.3"]})
    monkeypatch.setattr(executor, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match="sandbox pandas is wrong"):
        with sandbox_environment():
            pytest.fail("mismatched environment must not reach inference")
    assert executor._RUNNER == original


def test_wrapper_versions_resume_and_in_process_restoration(tmp_path):
    names = ("build_generator_messages", "generator_prompt_info", "run_repair", "AGENT_PROMPT_VERSION", "config_signature")
    originals = {name: getattr(runner_stage2, name) for name in names}
    args = ["--preset", "gold_contract", "--backend", "pipeline_check", "--splits", "dev",
            "--tasks", "adult_col_006", "--out", str(tmp_path)]
    old = main([*args, "--generator-prompt", "b2-v1", "--agent-prompt", "a1-v4"])
    revised = main([*args, "--generator-prompt", "b2-v2", "--agent-prompt", "a1-v5"])
    assert old != revised
    info = json.loads((revised / "run_info.json").read_text())
    assert info["valid_result"] is False
    assert info["config"]["generator_prompt"]["version"] == "b2-v2"
    assert info["config"]["agent_prompt_version"] == "a1-v5"
    assert info["config"]["sandbox_environment"]["pandas"] == "2.3.3"
    assert all(getattr(runner_stage2, name) == obj for name, obj in originals.items())
    # Change a budget while retaining the same name: the existing runner must refuse the resume.
    with pytest.raises(SystemExit, match="another configuration"):
        main([*args, "--generator-prompt", "b2-v2", "--agent-prompt", "a1-v5", "--max-attempts", "2"])
    assert all(getattr(runner_stage2, name) == obj for name, obj in originals.items())


def test_development_notebook_has_no_saved_outputs_and_all_code_compiles():
    notebook = json.loads((ROOT / "notebooks/kaggle/stage2_prompt_dev_kaggle.ipynb").read_text())
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), f"cell-{i}", "exec")
