"""Atomic per-task records, self-checking bundles, and a strict CPU-only collector."""
import csv
import io
import json
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

from . import metrics
from .config import EXPERIMENT, MODELS, SYSTEMS, canonical, digest, generator_version, validate_plan


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False).encode() + b"\n")


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def bundle_name(preset):
    return f"stage2_{SYSTEMS.index(preset) + 1:02d}_{preset}_bundle.zip"


def jsonl(data):
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def rows_bytes(rows):
    return b"".join(canonical(row) + b"\n" for row in rows)


def validate_rows(rows, manifest, require_complete=False):
    plan, preset = manifest["plan"], manifest["preset"]
    expected = {(tid, 0) for tid in plan["selected_test_ids"]}
    keys = [(r["task_id"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("Duplicate or unexpected task/seed records")
    if require_complete and set(keys) != expected:
        raise ValueError(f"Incomplete {preset}: {len(keys)}/{len(expected)} records")
    for row in rows:
        checks = dict(system=preset, preset=preset, split="test", agent_prompt_version="a1-v4",
                      generator_prompt=generator_version(preset), generator_model=MODELS["generator"]["id"],
                      agent_model=MODELS["agent"]["id"], experiment=EXPERIMENT)
        if any(row.get(k) != v for k, v in checks.items()):
            raise ValueError(f"Mislabeled record: {row['task_id']}")
        if row["outcome"] not in ("ACCEPT", "REJECT", "ESCALATE") or type(row["final_correct"]) is not bool:
            raise ValueError("Invalid task outcome")
        if row["final_correct"] and row["outcome"] != "ACCEPT":
            raise ValueError("Only accepted outputs can be final-correct")
        if row.get("stage_ended") == "parse" and (row["outcome"] != "ESCALATE" or row["final_correct"]):
            raise ValueError("Invalid parser escalation")
    return len(keys) == len(expected)


def task_payloads(directory, manifest):
    payloads = []
    for path in sorted((directory / "tasks").glob("*.json")):
        data = json.loads(path.read_text())
        row = data["record"]
        if path.name != f"{row['task_id']}__{row['seed']}.json":
            raise ValueError("Task filename and record identity differ")
        payloads.append(data)
    payloads.sort(key=lambda p: (p["record"]["task_id"], p["record"]["seed"]))
    validate_rows([p["record"] for p in payloads], manifest)
    return payloads


def export_arm(directory, destination, status="running"):
    directory, destination = Path(directory), Path(destination)
    manifest = json.loads((directory / "arm_manifest.json").read_text())
    items = task_payloads(directory, manifest)
    rows = [p["record"] for p in items]
    traces = [p["trace"] for p in items if p["trace"] is not None]
    coverage_complete = validate_rows(rows, manifest)
    successful = coverage_complete and status == "complete"
    atomic_write(directory / "runs.jsonl", rows_bytes(rows))
    atomic_write(directory / "traces.jsonl", rows_bytes(traces))
    write_json(directory / "metrics.json", metrics.summarize(rows))
    info = dict(system=manifest["preset"], experiment=EXPERIMENT, plan_hash=digest(canonical(manifest["plan"])),
                valid_result=manifest["plan"]["backend"] == "hf", n_runs=len(rows), complete=successful)
    write_json(directory / "run_info.json", info)
    files = {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "bundle_info.json" and not p.name.endswith(".tmp")}
    progress = dict(experiment=EXPERIMENT, preset=manifest["preset"], status=status,
                    complete=successful, valid_result=info["valid_result"], completed_records=len(rows),
                    expected_records=len(manifest["plan"]["selected_test_ids"]),
                    files={name: digest(data) for name, data in files.items()})
    write_json(directory / "bundle_info.json", progress)
    files["bundle_info.json"] = (directory / "bundle_info.json").read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    prefix = f"stage2_clean_v1/{manifest['preset']}/"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(prefix + name, data)
    temp.replace(destination)
    return progress


def load_bundle(path):
    path = Path(path)
    if path.is_dir():
        if any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError("Archive links are not supported")
        files = {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    else:
        files = {}
        with zipfile.ZipFile(path) as archive:
            seen, preset = set(), None
            for entry in archive.infolist():
                name = PurePosixPath(entry.filename)
                if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Unsafe archive path or link")
                if entry.filename in seen:
                    raise ValueError("Duplicate archive member")
                seen.add(entry.filename)
                if entry.is_dir():
                    continue
                if len(name.parts) < 3 or name.parts[0] != "stage2_clean_v1" or name.parts[1] not in SYSTEMS:
                    raise ValueError("This ZIP is not a standalone Stage 2 clean-workflow bundle")
                if preset is not None and preset != name.parts[1]:
                    raise ValueError("Multiple systems in one arm ZIP")
                preset = name.parts[1]
                files[PurePosixPath(*name.parts[2:]).as_posix()] = archive.read(entry)
    progress = json.loads(files["bundle_info.json"])
    if not path.is_dir() and progress.get("preset") != preset:
        raise ValueError("Archive folder and system metadata disagree")
    hashes = {name: digest(data) for name, data in files.items() if name != "bundle_info.json"}
    if hashes != progress["files"]:
        raise ValueError(f"Bundle file checksums do not match: {path}")
    return files


def validate_bundle(files, allow_scripted=False):
    manifest = json.loads(files["arm_manifest.json"])
    preset, plan = manifest["preset"], manifest["plan"]
    if preset not in SYSTEMS:
        raise ValueError("Unknown system")
    validate_plan(plan, allow_scripted=allow_scripted)
    progress = json.loads(files["bundle_info.json"])
    if progress.get("preset") != preset or progress.get("experiment") != EXPERIMENT:
        raise ValueError("Bundle metadata does not match its manifest")
    if progress.get("status") != "complete" or progress.get("complete") is not True:
        raise ValueError(f"{preset} did not complete successfully; do not include partial results")
    rows = jsonl(files["runs.jsonl"])
    validate_rows(rows, manifest, require_complete=True)
    if progress["completed_records"] != len(rows) or progress["expected_records"] != len(plan["selected_test_ids"]):
        raise ValueError("Incorrect completion counts")
    info = json.loads(files["run_info.json"])
    expected_info = dict(system=preset, experiment=EXPERIMENT, plan_hash=digest(canonical(plan)),
                         valid_result=plan["backend"] == "hf", n_runs=len(rows), complete=True)
    if info != expected_info or progress.get("valid_result") != info["valid_result"]:
        raise ValueError("Invalid run metadata")
    items = [json.loads(data) for name, data in sorted(files.items()) if name.startswith("tasks/") and name.endswith(".json")]
    items.sort(key=lambda p: (p["record"]["task_id"], p["record"]["seed"]))
    if [p["record"] for p in items] != rows:
        raise ValueError("Atomic task records disagree with runs.jsonl")
    traces = jsonl(files["traces.jsonl"])
    if [p["trace"] for p in items if p["trace"] is not None] != traces:
        raise ValueError("Atomic task traces disagree with traces.jsonl")
    keys = [(t["task_id"], t["seed"]) for t in traces]
    expected_traces = {(r["task_id"], r["seed"]) for r in rows if r["stage_ended"] != "parse"}
    if len(keys) != len(set(keys)) or set(keys) != expected_traces:
        raise ValueError("Missing or duplicate task traces")
    if json.loads(files["metrics.json"]) != metrics.summarize(rows):
        raise ValueError("Stored metrics disagree with task records")
    if plan["backend"] == "hf":
        sandbox = json.loads(files["sandbox.json"])
        if sandbox.get("version") != "sb1" or sandbox.get("pandas") != "2.3.3" or sandbox.get("numpy") != "2.1.3":
            raise ValueError("Incorrect sandbox data-library versions")
        if sandbox.get("python") != plan["environment"]["python"]:
            raise ValueError("Sandbox and inference Python versions differ")
        events = jsonl(files.get("diagnostics.jsonl", b""))
        passed = {(e.get("device"), e.get("case")) for e in events if e.get("event") == "kernel_self_test_passed"}
        required = {(f"cuda:{d}", case) for d in (0, 1) for case in ("causal_prefill", "decode", "masked_prefill")}
        if not passed >= required:
            raise ValueError("GPU self-test diagnostics are incomplete")
        roles = {"generator"} if preset == "baseline1" else {"generator", "agent"}
        loads = {e.get("role"): e for e in events if e.get("event") == "model_loaded"}
        for role in roles:
            spec, loaded = MODELS[role], loads.get(role, {})
            if any(loaded.get(k) != value for k, value in dict(model=spec["id"], revision=spec["revision"],
                                                            device=f"cuda:{spec['device']}", four_bit=spec["four_bit"]).items()):
                raise ValueError(f"Missing/incorrect model load metadata: {role}")
            if role == "agent" and loaded.get("quantized_parameters", 0) <= 0:
                raise ValueError("Repair model has no verified four-bit parameters")
    return preset, plan, rows


def discover_inputs(root):
    root = Path(root)
    zipped = sorted(root.rglob("stage2_0[1-6]_*.zip"))
    expanded = sorted({p.parent for p in root.rglob("arm_manifest.json")})
    return zipped + expanded


def collect(paths, output, allow_scripted=False):
    """Validate every input before writing anything; exact duplicate ZIP/tree inputs are harmless."""
    arms, shared = {}, None
    for path in paths:
        files = load_bundle(path)
        preset, plan, rows = validate_bundle(files, allow_scripted=allow_scripted)
        if shared is None:
            shared = plan
        elif shared != plan:
            raise ValueError("The six runs used different configurations or environments")
        if preset in arms:
            if arms[preset][0] != files:
                raise ValueError(f"Conflicting copies of {preset}; attach only one completed version")
            continue
        arms[preset] = (files, rows)
    missing = sorted(set(SYSTEMS) - set(arms))
    if missing:
        raise ValueError(f"Missing complete system bundles: {missing}")
    report = dict(experiment=EXPERIMENT, complete=True, valid_result=shared["backend"] == "hf",
                  total_records=sum(len(arms[p][1]) for p in SYSTEMS),
                  systems={p: metrics.summarize(arms[p][1]) for p in SYSTEMS},
                  paired_final_correct={f"{a} vs {b}": metrics.pair(arms[a][1], arms[b][1])
                                        for i, a in enumerate(SYSTEMS) for b in SYSTEMS[i + 1:]})
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Collector output already exists; use an empty output folder")
    output.mkdir(parents=True, exist_ok=True)
    for preset, (files, _) in arms.items():
        for name, data in files.items():
            atomic_write(output / preset / name, data)
    write_json(output / "experiment_plan.json", shared)
    write_json(output / "stage2_metrics_test.json", report)
    stream = io.StringIO(newline="")
    fields = ["system", "n_runs", "final_accuracy", "no_repair_accuracy", "repair_gain", "repair_success",
              "silent_error_rate", "escalate", "parse_escalations", "mean_tokens"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for preset, summary in report["systems"].items():
        writer.writerow(dict(system=preset, **{k: summary[k] for k in fields if k != "system"}))
    atomic_write(output / "stage2_summary.csv", stream.getvalue().encode())
    destination = output.parent / "stage2_complete_results.zip"
    temp = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, "stage2_complete_results/" + path.relative_to(output).as_posix())
    temp.replace(destination)
    return report, destination
