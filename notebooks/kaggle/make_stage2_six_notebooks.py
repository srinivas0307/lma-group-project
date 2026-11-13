"""Generate six self-contained evaluation notebooks and an offline CPU collector."""
import base64
import hashlib
import json
import textwrap
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SYSTEMS = ["baseline1", "baseline2_repair", "baseline3_repair", "gold_contract",
           "proposed", "no_contract_to_generator"]


def cell(source, kind="code"):
    item = dict(cell_type=kind, id=hashlib.sha256((kind + source).encode()).hexdigest()[:12], metadata={},
                source=source.splitlines(keepends=True))
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def notebook(cells):
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                              language_info=dict(name="python", version="3.13.16"), stage2_workflow="stage2-clean-v1"))


def runtime_cell():
    directory = ROOT / "eval/stage2_clean"
    files = {p.name: p.read_text() for p in sorted(directory.iterdir()) if p.suffix in (".py", ".json")}
    raw = json.dumps(files, sort_keys=True).encode()
    checksum = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(zlib.compress(raw, level=9)).decode()
    source = '''# Shared runtime embedded identically in all seven notebooks. No code upload is needed.
import base64, hashlib, json, os, subprocess, sys, time, zipfile, zlib
from pathlib import Path
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
'''
    source += f'RUNTIME_SHA256 = {checksum!r}\nPAYLOAD = (\n'
    source += "\n".join("    " + repr(line) for line in textwrap.wrap(encoded, 100)) + "\n)\n"
    source += '''raw = zlib.decompress(base64.b64decode(PAYLOAD))
assert hashlib.sha256(raw).hexdigest() == RUNTIME_SHA256, "Runtime payload was modified"
RUNTIME = WORK / ("stage2_runtime_" + RUNTIME_SHA256[:12])
package = RUNTIME / "stage2_clean"
package.mkdir(parents=True, exist_ok=True)
for filename, source in json.loads(raw).items():
    assert Path(filename).name == filename
    target = package / filename
    if target.exists():
        assert target.read_text() == source, f"Runtime file was changed: {target}"
    else:
        target.write_text(source)
previous = sys.modules.get("stage2_clean")
if previous is not None:
    assert Path(previous.__file__).resolve().parent == package.resolve(), "Restart the kernel after switching notebook versions"
sys.path.insert(0, str(RUNTIME))
from stage2_clean.config import EXPERIMENT, REPO_URL, REVISION, SYSTEMS, verify_sources
from stage2_clean.artifacts import bundle_name, export_arm, load_bundle, validate_bundle
print("Workflow:", EXPERIMENT, "runtime:", RUNTIME_SHA256[:12])
'''
    return cell(source)


SETUP = '''# Fetch the frozen research source. The shared runtime lives outside this checkout.
REPO = WORK / "repo_stage2_clean_v1"
if not (REPO / ".git").is_dir():
    REPO.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(REPO)], check=True)
    subprocess.run(["git", "remote", "add", "origin", REPO_URL], cwd=REPO, check=True)
current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
if current.returncode or current.stdout.strip() != REVISION:
    subprocess.run(["git", "fetch", "--depth", "1", "origin", REVISION], cwd=REPO, check=True)
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO, check=True)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == REVISION
verify_sources(REPO)
PACKAGES = WORK / "stage2_clean_packages"
SETUP_LOG = WORK / "stage2_setup.log"
commands = [
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--target", str(PACKAGES),
     "-r", str(REPO / "requirements.txt")],
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "bitsandbytes==0.50.2", "peft==0.21.2"],
]
with SETUP_LOG.open("a") as log:
    for command in commands:
        result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        log.flush()
        if result.returncode:
            raise RuntimeError("Dependency installation failed:\\n" + SETUP_LOG.read_text()[-6000:])
# Only child processes import inference/data libraries; the notebook kernel stays lightweight.
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [RUNTIME, PACKAGES, REPO, REPO / "src"])),
           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
probe = subprocess.run([sys.executable, "-c", "import json; from stage2_clean.backend import environment; print(json.dumps(environment(), indent=2))"],
                       cwd=REPO, env=ENV, capture_output=True, text=True)
if probe.returncode:
    raise RuntimeError("Environment check failed before evaluation:\\n" + probe.stderr[-6000:])
print(probe.stdout)
print("Frozen source and environment verified. Setup details:", SETUP_LOG)
'''

RUN = '''# Run this system's 100 tasks. Every completed task produces a fresh atomic checkpoint ZIP.
OUT = WORK / "stage2_clean_v1"
ARM = OUT / PRESET
BUNDLE = WORK / bundle_name(PRESET)
LOG = WORK / (PRESET + ".log")
command = [sys.executable, "-m", "stage2_clean.runner", "--preset", PRESET,
           "--repo", str(REPO), "--out", str(OUT), "--packages", str(PACKAGES)]
failure = None
code = None
started = time.monotonic()
try:
    with LOG.open("a") as log:
        log.write("Command: " + json.dumps(command) + "\\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if line.startswith(("GPU:", "TASK START:", "TASK DONE:", "COMPLETE:")):
                    print(line, end="", flush=True)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        log.write(f"\\nExit: {code}; minutes: {(time.monotonic()-started)/60:.2f}\\n")
    if code:
        raise RuntimeError(f"Evaluation stopped. Full log: {LOG}\\n" + LOG.read_text()[-10000:])
except BaseException as error:
    failure = error
finally:
    if (ARM / "arm_manifest.json").exists():
        try:
            if LOG.exists():
                from stage2_clean.artifacts import atomic_write
                atomic_write(ARM / "console.log", LOG.read_bytes())
            progress = export_arm(ARM, BUNDLE, "complete" if code == 0 and failure is None else "failed")
            print("Saved:", progress["completed_records"], "/", progress["expected_records"],
                  "complete:", progress["complete"], "->", BUNDLE)
        except Exception as error:
            print("Final export failed; retain the last saved ZIP:", error)
            if failure is None: failure = error
    from IPython.display import FileLink, display
    previous_directory = Path.cwd()
    try:
        os.chdir(WORK)
        if BUNDLE.exists(): display(FileLink(BUNDLE.name))
        if LOG.exists(): display(FileLink(LOG.name))
    finally:
        os.chdir(previous_directory)
if failure is not None:
    raise failure
files = load_bundle(BUNDLE)
validate_bundle(files)
print("COMPLETE AND VALIDATED:", PRESET, "100 / 100. Download", BUNDLE.name)
print("Summary:", json.loads(files["metrics.json"]))
'''


def build():
    shared = runtime_cell()
    results = {}
    for index, preset in enumerate(SYSTEMS, 1):
        title = f"# Stage 2 — {index:02d}: {preset}\n\n"
        title += (
            "**Workflow: stage2-clean-v1. Run notebook 06 first.**\n\n"
            "Start a fresh Kaggle notebook with **Internet on** and **GPU T4 ×2**. "
            "No previous results, adapter weights, secrets or GitHub push are required. "
            "Use **Save Version → Save & Run All**. This notebook evaluates exactly one system on all 100 test tasks.\n\n"
            "A small attention-kernel self-test runs on both GPUs before model loading. "
            "The previously failing task `taxi_dedup_003` runs first and counts once in the evaluation. "
            "Task-start, task-completion and GPU-memory diagnostics are recorded. "
            "A full GPU run has not been validated locally; failure is never counted as a task rejection.\n\n"
            "Download this system's output ZIP only after **COMPLETE AND VALIDATED: 100 / 100**. "
            "Each completed task is saved automatically. Rerunning cells within the same session keeps matching "
            "completed work; a fresh session starts a fresh evaluation. These results form a new experiment "
            "with a uniform memory-efficient attention runtime. Keep old experiments separate.\n")
        config = f'# Only the preset differs between the six evaluation notebooks.\nPRESET = {preset!r}\n'
        ending = (
            "## Next\n\nDownload `" + f"stage2_{index:02d}_{preset}_bundle.zip" + "` and this executed notebook. "
            "After all six systems complete, attach their notebook outputs or six ZIPs to "
            "`stage2_collect_results.ipynb`. The collector needs **no GPU or Internet**. "
            "It checks all 600 records and recomputes the comparison.\n")
        results[f"stage2_{index:02d}_{preset}.ipynb"] = notebook(
            [cell(title, "markdown"), cell(config), shared, cell(SETUP), cell(RUN), cell(ending, "markdown")])
    collect_code = '''# Attach the six COMPLETED notebook outputs (or their six ZIP files) as Kaggle inputs.
from stage2_clean.artifacts import collect, discover_inputs
inputs = discover_inputs(Path("/kaggle/input"))
assert inputs, "Attach the six completed Stage 2 outputs as inputs before running this notebook"
print("Input bundles/directories:", *map(str, inputs), sep="\\n")
OUTPUT = WORK / ("stage2_collected_" + str(time.time_ns()))
report, archive = collect(inputs, OUTPUT)
assert report["complete"] and report["valid_result"] and report["total_records"] == 600
print("COMPLETE: all six systems, 600 / 600 verified task records")
for system, summary in report["systems"].items():
    print(system, "final accuracy:", summary["final_accuracy"], "repair gain:", summary["repair_gain"])
print("Full report:", OUTPUT / "stage2_metrics_test.json")
print("Spreadsheet-friendly summary:", OUTPUT / "stage2_summary.csv")
from IPython.display import FileLink, display
previous_directory = Path.cwd()
try:
    os.chdir(WORK)
    display(FileLink(archive.name))
finally:
    os.chdir(previous_directory)
'''
    results["stage2_collect_results.ipynb"] = notebook([
        cell("# Collect the six Stage 2 results\n\n**GPU: None. Internet: not required.** "
             "Attach all six completed notebook outputs or upload their ZIPs as inputs, then Run All. "
             "ZIPs and automatically unpacked uploads are both supported. Exact duplicate views of the same "
             "bundle are accepted; conflicting versions, missing tasks, changed settings and old experiment "
             "outputs are rejected. The shared code is included in this notebook.\n", "markdown"),
        shared, cell(collect_code),
        cell("Download **stage2_complete_results.zip**. It includes all task traces, runtime diagnostics, "
             "per-system results, confidence intervals, paired comparisons and `stage2_summary.csv`. "
             "Only an accepted output equal to its reference counts as final-correct.\n", "markdown")])
    return results


def main():
    notebooks = build()
    for name, value in notebooks.items():
        (HERE / name).write_text(json.dumps(value, indent=1) + "\n")
    destination = HERE / "stage2_six_notebooks.zip"
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in notebooks:
            archive.write(HERE / name, name)
        guide = ROOT / "docs/stage2_six_notebooks.md"
        if guide.is_file():
            archive.write(guide, "START_HERE.md")
    print("Generated", len(notebooks), "notebooks:", destination)


if __name__ == "__main__":
    main()
