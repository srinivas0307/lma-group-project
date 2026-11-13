"""Recovery preserves the frozen method and verifies process-local GPU placement."""
import json
import runpy
import hashlib
import os
import sys
import time
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "archive/notebooks/core/make_stage2_resume_notebook.py"
NOTEBOOK = ROOT / "archive/notebooks/core/stage2_test_resume_kaggle.ipynb"


def test_generated_notebook_retains_frozen_checks_and_compiles():
    builder = runpy.run_path(str(BUILDER))
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook == builder["make_notebook"]()
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    original = json.loads((NOTEBOOK.parent / "stage2_test_frozen_kaggle.ipynb").read_text())
    # Source/data checksums, package checks, method signature and export/resume helpers are exact copies.
    for index in range(2, 7):
        assert notebook["cells"][index] == original["cells"][index]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            compile("".join(cell["source"]), f"resume-cell-{index}", "exec")
    scope = {}
    exec("".join(notebook["cells"][1]["source"]), scope)
    assert scope["SYSTEMS"] == ["no_contract_to_generator"]
    assert scope["RESUME_BUNDLE"] == "auto"


@pytest.mark.parametrize("map_state", ["present", "missing", "none"])
def test_placement_assigns_distinct_devices_and_preserves_loading_options(map_state):
    builder = runpy.run_path(str(BUILDER))
    scope = {"__name__": "placement_test"}
    exec(builder["BOOTSTRAP"], scope)
    calls, records = [], []
    quantization = object()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, *args, **kwargs):
            calls.append((model_id, args, kwargs))
            gpu = kwargs["device_map"][""]
            model = SimpleNamespace(
                parameters=lambda: iter([SimpleNamespace(device=f"cuda:{gpu}")]),
            )
            if map_state != "missing":
                model.hf_device_map = {"": gpu} if map_state == "present" else None
            return model

    scope["install_placement"](FakeAutoModel, {"repair": 0, "generator": 1}, records.append)
    FakeAutoModel.from_pretrained("repair", dtype="float16", quantization_config=quantization, device_map="auto")
    FakeAutoModel.from_pretrained("generator", dtype="float16", device_map="auto")
    assert calls[0][2] == dict(dtype="float16", quantization_config=quantization, device_map={"": 0})
    assert calls[1][2] == dict(dtype="float16", device_map={"": 1})
    assert [r["parameter_devices"] for r in records] == [["cuda:0"], ["cuda:1"]]
    assert [r["hf_device_map"] for r in records] == (
        [{"": "0"}, {"": "1"}] if map_state == "present" else [{}, {}]
    )
    with pytest.raises(RuntimeError, match="Unexpected model"):
        FakeAutoModel.from_pretrained("wrong-model")
    assert len(calls) == 2


def test_placement_refuses_unexpected_actual_device():
    builder = runpy.run_path(str(BUILDER))
    scope = {"__name__": "placement_test"}
    exec(builder["BOOTSTRAP"], scope)
    records = []

    class WrongDevice:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cuda:0")]))

    scope["install_placement"](WrongDevice, {"generator": 1}, records.append)
    with pytest.raises(RuntimeError, match="Placement check failed"):
        WrongDevice.from_pretrained("generator")
    assert records == []


def test_resume_input_discovery_handles_dataset_unpacking_and_ambiguity(tmp_path):
    builder = runpy.run_path(str(BUILDER))
    provided = tmp_path / "provided.zip"
    provided.write_bytes(b"placeholder")
    scope = dict(Path=Path, zipfile=zipfile, RESUME_BUNDLE=str(provided))
    exec(builder["DISCOVER_INPUT"], scope)
    inputs = tmp_path / "inputs"
    extracted = inputs / "uploaded-dataset/results/stage2_test_frozen"
    extracted.mkdir(parents=True)
    (extracted / "frozen_plan.json").write_text('{"fixture": true}')
    records = extracted / "partial/runs.jsonl"
    records.parent.mkdir()
    records.write_bytes(b'{"task_id":"example"}\n')
    recovered = scope["find_resume_bundle"](inputs, tmp_path)
    with zipfile.ZipFile(recovered) as archive:
        assert archive.read("results/stage2_test_frozen/partial/runs.jsonl") == records.read_bytes()
        assert json.loads(archive.read("results/stage2_test_frozen/frozen_plan.json")) == {"fixture": True}
    zipped = inputs / "stage2_test_bundle.zip"
    zipped.write_bytes(provided.read_bytes())
    assert scope["find_resume_bundle"](inputs, tmp_path) == str(zipped)
    duplicate = inputs / "another/stage2_test_bundle.zip"
    duplicate.parent.mkdir()
    duplicate.write_bytes(provided.read_bytes())
    with pytest.raises(AssertionError, match="Several checkpoint ZIPs"):
        scope["find_resume_bundle"](inputs, tmp_path)


def setup_scope(tmp_path):
    """Exercise the real setup cell in a persistent notebook-like namespace."""
    builder = runpy.run_path(str(BUILDER))
    scope = dict(json=json, hashlib=hashlib, os=os, sys=sys, time=time, Path=Path)
    notebook = builder["make_notebook"]()
    exec("".join(notebook["cells"][1]["source"]), scope)
    out = tmp_path / "results"
    out.mkdir()
    checkpoint = tmp_path / "input.zip"
    checkpoint.write_bytes(b"already validated fixture checkpoint")
    for preset in scope["ALL_SYSTEMS"]:
        directory = out / preset
        directory.mkdir()
        count = 1 if preset == "no_contract_to_generator" else 2
        (directory / "runs.jsonl").write_text("".join(
            json.dumps(dict(task_id=tid, seed=0)) + "\n" for tid in ["first", "second"][:count]))

    def coverage(preset):
        rows = (out / preset / "runs.jsonl").read_text().splitlines()
        return dict(records=len(rows), complete=len(rows) == 2)

    scope.update(OUT=out, WORK=str(tmp_path), RESUME_BUNDLE=str(checkpoint),
                 TEST_IDS=["first", "second"], COMMIT="fixture", VERSIONS={}, ENV={},
                 GPU_SPLIT_BOOTSTRAP=builder["BOOTSTRAP"], coverage=coverage,
                 build_command=lambda preset: [sys.executable, "eval/runner_stage2_prompts.py", "--preset", preset])
    return builder, scope


def test_setup_can_be_rerun_without_recursive_or_duplicate_wrappers(tmp_path):
    builder, scope = setup_scope(tmp_path)
    original_command = scope["build_command"]("no_contract_to_generator")
    for _ in range(3):
        exec(builder["SETUP"], scope)
        command = scope["build_command"]("no_contract_to_generator")
        assert command == [original_command[0], str(scope["BOOTSTRAP_PATH"]), *original_command[1:]]
        scope["verify_preserved_records"]()
    (scope["OUT"] / "no_contract_to_generator/runs.jsonl").write_text("changed history")
    with pytest.raises(AssertionError, match="Existing records were changed"):
        scope["verify_preserved_records"]()


def test_setup_refuses_missing_completed_systems_before_model_loading(tmp_path):
    builder, scope = setup_scope(tmp_path)
    (scope["OUT"] / "baseline1/runs.jsonl").write_text("")
    with pytest.raises(AssertionError, match="other five systems"):
        exec(builder["SETUP"], scope)
    assert not (tmp_path / "stage2_gpu_split_bootstrap.py").exists()


@pytest.mark.parametrize("self_test_ok", [True, False])
def test_bootstrap_entry_point_loads_both_models_without_optional_metadata(tmp_path, monkeypatch, self_test_ok):
    builder = runpy.run_path(str(BUILDER))
    scope = {"__name__": "bootstrap_test"}
    exec(builder["BOOTSTRAP"], scope)
    # Kernel behavior is tested separately; this test exercises bootstrap/runner integration.
    def configure(torch, record):
        if not self_test_ok:
            raise RuntimeError("GPU self-test failed before weights")
        record(dict(event="stub_self_test_passed"))

    scope["configure_attention"] = configure
    calls = []

    class AutoModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            gpu = kwargs["device_map"][""]
            calls.append((name, kwargs))
            # Match the user's failing condition: no hf_device_map attribute.
            return SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=f"cuda:{gpu}")]))

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = AutoModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    provenance = tmp_path / "placement.json"
    provenance.write_text(json.dumps(dict(model_gpu_ordinals={"repair": 0, "generator": 1}, model_loads=[])))
    monkeypatch.setenv("STAGE2_PLACEMENT_RECORD", str(provenance))
    runner = tmp_path / "runner.py"
    runner.write_text(
        'import sys\nfrom transformers import AutoModelForCausalLM\n'
        'assert sys.argv[1:] == ["--resume", "38"]\n'
        'AutoModelForCausalLM.from_pretrained("repair", dtype="float16", quantization_config="nf4")\n'
        'AutoModelForCausalLM.from_pretrained("generator", dtype="float16")\n')
    monkeypatch.setattr(sys, "argv", ["bootstrap.py", str(runner), "--resume", "38"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    if not self_test_ok:
        with pytest.raises(RuntimeError, match="GPU self-test failed before weights"):
            scope["main"]()
        assert calls == []
        assert json.loads(provenance.read_text())["model_loads"] == []
        return
    scope["main"]()
    records = json.loads(provenance.read_text())["model_loads"]
    assert [r["parameter_devices"] for r in records] == [["cuda:0"], ["cuda:1"]]
    assert [r["hf_device_map"] for r in records] == [{}, {}]
    assert calls[0][1] == dict(dtype="float16", quantization_config="nf4", device_map={"": 0})
    assert calls[1][1] == dict(dtype="float16", device_map={"": 1})


@pytest.mark.parametrize("scenario", ["success", "model_failure", "export_failure", "history_changed"])
def test_execution_from_download_directory_preserves_checkpoint_and_original_errors(tmp_path, monkeypatch, scenario):
    builder, scope = setup_scope(tmp_path)
    exec(builder["SETUP"], scope)
    repo = tmp_path / "repo"
    repo.mkdir()
    logs = scope["OUT"] / "logs"
    logs.mkdir()
    bundle = tmp_path / "output.zip"
    exports, launches = [], []

    def export(status):
        if scenario == "export_failure" and exports:
            raise ValueError("simulated export failure")
        exports.append(status)
        bundle.write_text(json.dumps(exports))
        return dict(status=status)

    class Process:
        stdout = iter(["Verified GPU placement: fixture\n"])

        def wait(self):
            partial = scope["OUT"] / "no_contract_to_generator/runs.jsonl"
            if scenario == "success":
                with partial.open("a") as stream:
                    stream.write(json.dumps(dict(task_id="second", seed=0)) + "\n")
                return 0
            if scenario == "history_changed":
                partial.write_text("changed history")
            return 1

    def popen(command, **kwargs):
        assert kwargs["cwd"] == repo
        assert Path.cwd() == repo
        launches.append(command)
        return Process()

    ipython = ModuleType("IPython")
    ipython.__path__ = []
    display = ModuleType("IPython.display")
    display.display = lambda value: None
    display.FileLink = lambda value: value
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.chdir(tmp_path)  # The final download cell leaves the notebook here.
    scope.update(REPO=repo, LOGS=logs, BUNDLE=bundle, export_bundle=export,
                 system_name=lambda preset: preset,
                 subprocess=SimpleNamespace(Popen=popen, PIPE=-1, STDOUT=-2))
    execution = "".join(builder["make_notebook"]()["cells"][9]["source"])
    if scenario == "success":
        exec(execution, scope)
        assert exports == ["partial", "partial", "partial"]
        scope["verify_preserved_records"]()
    else:
        with pytest.raises(RuntimeError, match="no_contract_to_generator failed"):
            exec(execution, scope)
        assert exports == (["partial", "failed"] if scenario == "model_failure" else ["partial"])
    assert len(launches) == 1
    assert json.loads(bundle.read_text()) == exports  # Last safe checkpoint is retained on cleanup failure.
    assert Path.cwd() == repo
