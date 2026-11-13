"""CPU end-to-end coverage of six arms, atomic failures, strict aggregation and notebook packaging."""
import copy
import json
import runpy
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from eval.stage2_clean import artifacts, config, metrics, runner
from eval.stage2_metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = Path(pd.__file__).resolve().parent.parent
TASKS = ["titanic_filt_003", "airbnb_miss_004"]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("six-systems")
    out = root / "arms"
    for preset in config.SYSTEMS:
        runner.run(preset, ROOT, out, PACKAGES, backend="pipeline_check", task_ids=TASKS)
    return root, [root / artifacts.bundle_name(p) for p in config.SYSTEMS]


def reseal(files):
    progress = json.loads(files["bundle_info.json"])
    progress["files"] = {name: config.digest(data) for name, data in files.items() if name != "bundle_info.json"}
    files["bundle_info.json"] = config.canonical(progress)
    return files


def write_fixture(path, preset, files):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(f"stage2_clean_v1/{preset}/{name}", data)


def test_all_six_real_evaluation_paths_and_collector(completed, tmp_path):
    root, paths = completed
    for preset, path in zip(config.SYSTEMS, paths):
        files = artifacts.load_bundle(path)
        actual, plan, rows = artifacts.validate_bundle(files, allow_scripted=True)
        assert actual == preset and len(rows) == 2
        assert metrics.summarize(rows) == summarize(rows)
        with pytest.raises(ValueError, match="not model-quality"):
            artifacts.validate_bundle(files)
    proposed = artifacts.jsonl(artifacts.load_bundle(paths[4])["runs.jsonl"])
    assert any(r["task_id"] == "airbnb_miss_004" and r["stage_ended"] == "parse" for r in proposed)
    # Includes the duplicate ZIP + unpacked view that Kaggle notebook inputs expose.
    report, archive = artifacts.collect(paths + [paths[0], root / "arms/baseline1"], tmp_path / "collected",
                                        allow_scripted=True)
    assert report["complete"] and report["total_records"] == 12
    assert report["valid_result"] is False
    assert len(report["paired_final_correct"]) == 15
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert "stage2_complete_results/stage2_summary.csv" in bundle.namelist()
    discovered = artifacts.discover_inputs(root)
    assert set(paths) <= set(discovered)


def test_preserves_frozen_evaluator_behavior(completed, tmp_path):
    from eval.runner_stage2_prompts import main as legacy

    _, paths = completed
    # Exercise all six routes with the same scripted outputs against the prior evaluator.
    for preset, path in zip(config.SYSTEMS, paths):
        argv = ["--preset", preset, "--backend", "pipeline_check", "--out", str(tmp_path),
                "--tasks", *TASKS, "--generator-prompt", "b2-v2", "--agent-prompt", "a1-v4"]
        if preset in config.PARSES:
            argv += ["--parses", str(ROOT / config.PARSES[preset])]
        directory = legacy(argv)
        old = artifacts.jsonl((directory / "runs.jsonl").read_bytes())
        new = artifacts.jsonl(artifacts.load_bundle(path)["runs.jsonl"])
        ignored = {"system", "experiment", "latency_s"}
        normalize = lambda rows: sorted([{k: v for k, v in r.items() if k not in ignored} for r in rows],
                                       key=lambda r: r["task_id"])
        assert normalize(old) == normalize(new)


def test_infrastructure_failure_never_becomes_a_task_outcome(tmp_path, monkeypatch):
    original = runner.evaluate
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[0]["id"])
        if len(calls) == 2:
            raise RuntimeError("simulated GPU infrastructure failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate", fail_second)
    with pytest.raises(RuntimeError, match="simulated GPU"):
        runner.run("baseline1", ROOT, tmp_path / "arms", PACKAGES, "pipeline_check", TASKS)
    path = tmp_path / artifacts.bundle_name("baseline1")
    files = artifacts.load_bundle(path)
    assert len(artifacts.jsonl(files["runs.jsonl"])) == 1
    assert len([name for name in files if name.startswith("tasks/")]) == 1
    info = json.loads(files["bundle_info.json"])
    assert info["status"] == "failed" and info["complete"] is False
    with pytest.raises(ValueError, match="did not complete"):
        artifacts.validate_bundle(files, allow_scripted=True)
    previous = files["runs.jsonl"]
    monkeypatch.setattr(runner, "evaluate", original)
    runner.run("baseline1", ROOT, tmp_path / "arms", PACKAGES, "pipeline_check", TASKS)
    _, _, rows = artifacts.validate_bundle(artifacts.load_bundle(path), allow_scripted=True)
    assert len(rows) == 2 and artifacts.jsonl(previous)[0] in rows


@pytest.mark.parametrize("damage", ["duplicate", "missing_trace", "wrong_label", "changed_runtime", "partial"])
def test_rejects_inconsistent_results(completed, damage):
    _, paths = completed
    files = copy.deepcopy(artifacts.load_bundle(paths[0]))
    if damage == "duplicate":
        rows = artifacts.jsonl(files["runs.jsonl"])
        files["runs.jsonl"] = artifacts.rows_bytes(rows + [rows[0]])
    elif damage == "missing_trace":
        files["traces.jsonl"] = b""
    elif damage == "wrong_label":
        rows = artifacts.jsonl(files["runs.jsonl"])
        rows[0]["generator_model"] = "wrong-model"
        files["runs.jsonl"] = artifacts.rows_bytes(rows)
    elif damage == "changed_runtime":
        manifest = json.loads(files["arm_manifest.json"])
        manifest["plan"]["runtime"]["backend.py"] = "changed"
        files["arm_manifest.json"] = config.canonical(manifest)
    else:
        progress = json.loads(files["bundle_info.json"])
        progress["complete"], progress["status"] = False, "failed"
        files["bundle_info.json"] = config.canonical(progress)
    with pytest.raises(ValueError):
        artifacts.validate_bundle(reseal(files), allow_scripted=True)


def test_collector_rejects_missing_and_environment_mismatch_before_writes(completed, tmp_path):
    _, paths = completed
    with pytest.raises(ValueError, match="Missing complete"):
        artifacts.collect(paths[:-1], tmp_path / "missing", allow_scripted=True)
    assert not (tmp_path / "missing").exists()
    files = artifacts.load_bundle(paths[-1])
    manifest = json.loads(files["arm_manifest.json"])
    manifest["plan"]["environment"]["different"] = True
    files["arm_manifest.json"] = config.canonical(manifest)
    info = json.loads(files["run_info.json"])
    info["plan_hash"] = config.digest(config.canonical(manifest["plan"]))
    files["run_info.json"] = config.canonical(info)
    conflicting = tmp_path / "different.zip"
    write_fixture(conflicting, config.SYSTEMS[-1], reseal(files))
    with pytest.raises(ValueError, match="different configurations or environments"):
        artifacts.collect(paths[:-1] + [conflicting], tmp_path / "mixed", allow_scripted=True)
    assert not (tmp_path / "mixed").exists()


def test_checksum_and_archive_path_checks(completed, tmp_path):
    _, paths = completed
    files = artifacts.load_bundle(paths[0])
    files["runs.jsonl"] += b"{}\n"
    corrupt = tmp_path / "corrupt.zip"
    write_fixture(corrupt, "baseline1", files)
    with pytest.raises(ValueError, match="checksums"):
        artifacts.load_bundle(corrupt)
    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("stage2_clean_v1/baseline1/../../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        artifacts.load_bundle(malicious)
    if (ROOT / "archive/stage2/test/root_checkpoint/stage2_test_bundle.zip").exists():
        with pytest.raises(ValueError, match="not a standalone"):
            artifacts.load_bundle(ROOT / "archive/stage2/test/root_checkpoint/stage2_test_bundle.zip")


def test_generated_notebooks_are_self_contained_and_share_one_runtime():
    builder = runpy.run_path(str(ROOT / "notebooks/kaggle/make_stage2_six_notebooks.py"))
    generated = builder["build"]()
    assert len(generated) == 7
    runtime_sources = []
    for name, notebook in generated.items():
        assert json.loads((ROOT / "notebooks/kaggle" / name).read_text()) == notebook
        ids = [c["id"] for c in notebook["cells"]]
        assert len(set(ids)) == len(ids)
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code":
                assert not cell["outputs"] and cell["execution_count"] is None
                compile(source, name, "exec")
            if "PAYLOAD = (" in source:
                runtime_sources.append(source)
        if name != "stage2_collect_results.ipynb":
            namespace = {}
            exec("".join(notebook["cells"][1]["source"]), namespace)
            assert namespace["PRESET"] in config.SYSTEMS
    assert len(runtime_sources) == 7 and len(set(runtime_sources)) == 1
    assert config.ORDER[0] == "taxi_dedup_003" and set(config.ORDER) == set(config.TEST_IDS)
    config.verify_sources(ROOT)
