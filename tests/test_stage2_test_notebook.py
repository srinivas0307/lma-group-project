"""The final notebook routes all six arms correctly and safely resumes outputs."""
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
import pytest

from eval.runner_stage2_prompts import main

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "archive/notebooks/core/stage2_test_frozen_kaggle.ipynb"


def namespace(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text())
    ns = dict(json=json, sys=sys, Path=Path, PurePosixPath=PurePosixPath, stat=stat, zipfile=zipfile)
    config = next("".join(c["source"]) for c in notebook["cells"] if "ALL_SYSTEMS =" in "".join(c["source"]))
    exec(compile(config, "notebook-config", "exec"), ns)
    helpers = next("".join(c["source"]) for c in notebook["cells"] if "def build_command" in "".join(c["source"]))
    exec(compile(helpers, "notebook-helpers", "exec"), ns)
    ns.update(OUT=tmp_path / "results", CACHE=tmp_path / "cache",
              PACKAGES=Path(pd.__file__).resolve().parent.parent, TEST_IDS=["adult_col_001"])
    ns["FROZEN_PLAN"] = {k: ns[k.upper()] for k in ("max_attempts", "max_tool_calls", "token_budget")}
    return ns


def test_notebook_is_unexecuted_and_all_code_compiles():
    notebook = json.loads(NOTEBOOK.read_text())
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            compile("".join(cell["source"]), f"cell-{i}", "exec")


def test_six_system_commands_and_actual_pipeline_metadata(tmp_path):
    ns = namespace(tmp_path)
    ns["BACKEND"] = "pipeline_check"  # Test harness only; the actual notebook is locked to hf.
    for preset in ns["ALL_SYSTEMS"]:
        command = ns["build_command"](preset)
        assert command[command.index("--splits") + 1] == "test"
        assert command[command.index("--agent-prompt") + 1] == "a1-v4"
        assert command[command.index("--generator-prompt") + 1] == "b2-v2"
        if preset in ("proposed", "no_contract_to_generator", "baseline3_repair"):
            expected = ns["B3_PARSES"] if preset == "baseline3_repair" else ns["FT_PARSES"]
            assert command[command.index("--parses") + 1] == expected
        else:
            assert "--parses" not in command
        out = main(command[2:] + ["--tasks", "adult_col_001"])
        assert out.name == ns["system_name"](preset)
        assert ns["coverage"](preset) == dict(records=1, complete=True)
        info = json.loads((out / "run_info.json").read_text())
        assert info["valid_result"] is False
    # Exercise the actual export helper, including the metrics subprocess and ZIP layout.
    ns.update(COMMIT="scripted-test", ENV=dict(os.environ), SYSTEMS=ns["ALL_SYSTEMS"],
              VERSIONS={}, BUNDLE=tmp_path / "stage2_test_bundle.zip", subprocess=subprocess)
    (ns["OUT"] / "frozen_plan.json").write_text(json.dumps(ns["FROZEN_PLAN"]))
    progress = ns["export_bundle"]("partial")
    assert progress["experiment_complete"] and progress["completed_records"] == 6
    with zipfile.ZipFile(ns["BUNDLE"]) as archive:
        prefix = "results/stage2_test_frozen/"
        assert prefix + "bundle_info.json" in archive.namelist()
        assert prefix + "stage2_metrics_test.json" in archive.namelist()
        report = json.loads(archive.read(prefix + "stage2_metrics_test.json"))
        assert len(report["not_valid_results"]) == 6  # Scripted outputs cannot masquerade as real results.
    # A mislabeled or different agent record must not be counted as a completed arm.
    runs = ns["OUT"] / ns["system_name"]("proposed") / "runs.jsonl"
    row = json.loads(runs.read_text())
    row["agent_prompt_version"] = "a1-v5"
    runs.write_text(json.dumps(row) + "\n")
    with pytest.raises(AssertionError, match="Wrong repair prompt"):
        ns["coverage"]("proposed")


def test_restore_checks_plan_paths_and_does_not_overwrite(tmp_path):
    ns = namespace(tmp_path)
    prefix = "results/stage2_test_frozen/"
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr(prefix + "frozen_plan.json", json.dumps(ns["FROZEN_PLAN"]))
        z.writestr(prefix + "logs/example.log", "preserved")
    ns["restore_bundle"](good)
    assert (ns["OUT"] / "logs/example.log").read_text() == "preserved"
    (ns["OUT"] / "logs/example.log").write_text("newer data")
    with pytest.raises(AssertionError, match="Conflicting resume file"):
        ns["restore_bundle"](good)
    assert (ns["OUT"] / "logs/example.log").read_text() == "newer data"
    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as z:
        z.writestr(prefix + "frozen_plan.json", json.dumps(ns["FROZEN_PLAN"]))
        z.writestr(prefix + "../escape.txt", "must not write")
    with pytest.raises(AssertionError, match="Unsafe archive path"):
        ns["restore_bundle"](malicious)
    assert not (ns["OUT"].parent / "escape.txt").exists()
    wrong = tmp_path / "wrong-plan.zip"
    with zipfile.ZipFile(wrong, "w") as z:
        z.writestr(prefix + "frozen_plan.json", json.dumps({"wrong": True}))
    with pytest.raises(AssertionError, match="different experiment"):
        ns["restore_bundle"](wrong)
