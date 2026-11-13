"""Build the self-contained GPU-placement recovery notebook from the frozen notebook."""
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BOOTSTRAP = '''"""Process-local GPU placement; the frozen repository is never edited."""
import json
import os
import runpy
import sys
from pathlib import Path


def install_efficient_attention(functional, kernel_context, efficient_backend, record):
    """Expand grouped K/V heads explicitly, then require the fused CUDA kernel."""
    original = functional.scaled_dot_product_attention
    reported = set()

    def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                  *, scale=None, enable_gqa=False):
        if not str(query.device).startswith("cuda:"):
            return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        original_key_shape = list(key.shape)
        if enable_gqa:
            heads, kv_heads = query.shape[-3], key.shape[-3]
            if kv_heads != value.shape[-3] or heads % kv_heads:
                raise ValueError("Invalid grouped-query head counts")
            # This is the expansion in PyTorch's documented GQA definition.
            # Native enable_gqa=True cannot use the efficient CUDA kernel in torch 2.11.
            key = key.repeat_interleave(heads // kv_heads, dim=-3)
            value = value.repeat_interleave(heads // kv_heads, dim=-3)
        stage = "prefill" if query.shape[-2] > 1 else "decode"
        signature = (str(query.device), query.shape[-3], stage, attn_mask is not None)
        details = dict(device=str(query.device), query_shape=list(query.shape),
                       original_key_shape=original_key_shape, expanded_key_shape=list(key.shape),
                       dtype=str(query.dtype), native_gqa_expanded=bool(enable_gqa),
                       stage=stage, has_mask=attn_mask is not None, is_causal=bool(is_causal),
                       backend="EFFICIENT_ATTENTION", math_fallback_allowed=False)
        if signature not in reported:
            record(dict(event="attention_shape", **details))
            reported.add(signature)
        try:
            with kernel_context(efficient_backend):
                return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                is_causal=is_causal, scale=scale, enable_gqa=False)
        except Exception as error:
            record(dict(event="attention_failure", error=str(error), **details))
            raise

    functional.scaled_dot_product_attention = attention
    return original, attention


def configure_attention(torch, record):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original, attention = install_efficient_attention(
        torch.nn.functional, sdpa_kernel, SDPBackend.EFFICIENT_ATTENTION, record)
    # Test the actual Kaggle kernels before downloading/loading model weights.
    # Tiny math-reference calls are confined to this self-test, not model inference.
    with torch.inference_mode(), torch.random.fork_rng(devices=[0, 1]):
        for device in ("cuda:0", "cuda:1"):
            torch.manual_seed(0)
            query = torch.randn((1, 4, 32, 128), device=device, dtype=torch.float16)
            key = torch.randn((1, 2, 32, 128), device=device, dtype=torch.float16)
            value = torch.randn_like(key)
            mask = torch.zeros((32, 32), device=device, dtype=torch.float16)
            mask[:, -1] = float("-inf")
            cases = [("causal_prefill", query, None, True),
                     ("decode", query[:, :, :1, :], None, False),
                     ("masked_prefill", query, mask, False)]
            for name, q, bias, causal in cases:
                actual = attention(q, key, value, attn_mask=bias, is_causal=causal, enable_gqa=True)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = original(q, key, value, attn_mask=bias, is_causal=causal, enable_gqa=True)
                torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
                torch.cuda.synchronize(device)
                record(dict(event="kernel_self_test_passed", device=device, case=name,
                            max_absolute_error=float((actual - expected).abs().max().item()),
                            rtol=0.005, atol=0.005))
            del query, key, value, mask, cases, q, bias, actual, expected
    torch.cuda.empty_cache()


def install_placement(model_class, placements, record):
    original = model_class.from_pretrained

    def load(model_id, *args, **kwargs):
        if model_id not in placements:
            raise RuntimeError(f"Unexpected model in recovery process: {model_id}")
        device = placements[model_id]
        kwargs["device_map"] = {"": device}
        model = original(model_id, *args, **kwargs)
        actual = {str(parameter.device) for parameter in model.parameters()}
        if actual != {f"cuda:{device}"}:
            raise RuntimeError(f"Placement check failed for {model_id}: {actual}")
        # Single-device loading may omit this optional metadata.
        # The parameter-device check above establishes actual placement.
        reported_map = getattr(model, "hf_device_map", None) or {}
        record(dict(model=model_id, requested_device=f"cuda:{device}",
                    parameter_devices=sorted(actual),
                    hf_device_map={str(k): str(v) for k, v in reported_map.items()}))
        return model

    model_class.from_pretrained = staticmethod(load)


def main():
    import torch
    from transformers import AutoModelForCausalLM

    assert torch.cuda.is_available() and torch.cuda.device_count() == 2, "Requires GPU T4 x2"
    provenance = Path(os.environ["STAGE2_PLACEMENT_RECORD"])
    metadata = json.loads(provenance.read_text())

    def save(collection, item, prefix):
        metadata.setdefault(collection, []).append(item)
        temporary = provenance.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(provenance)
        print(prefix, json.dumps(item), flush=True)

    def record(item):
        save("model_loads", item, "Verified GPU placement:")

    def attention_record(item):
        save("attention_events", item, "Attention recovery:")

    configure_attention(torch, attention_record)
    install_placement(AutoModelForCausalLM, metadata["model_gpu_ordinals"], record)
    runner = Path(sys.argv[1]).resolve()
    # Match direct execution of the original runner (imports, argv, and source path).
    sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), *sys.argv[2:]]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
'''

DISCOVER_INPUT = '''# Attach the previous stage2_test_bundle.zip as a Kaggle input first.
def find_resume_bundle(input_root, work):
    candidates = sorted(input_root.rglob("stage2_test_bundle.zip"))
    if candidates:
        assert len(candidates) == 1, (
            "Several checkpoint ZIPs were found; set RESUME_BUNDLE to the exact path: "
            f"{[str(path) for path in candidates]}"
        )
        return str(candidates[0])
    # Dataset upload can unpack a ZIP automatically. Repackage its files, then apply
    # the same strict frozen-plan and archive checks used for an intact ZIP.
    plans = sorted(input_root.rglob("frozen_plan.json"))
    assert len(plans) == 1, (
        "Attach the previous output ZIP as an input. Could not identify one checkpoint: "
        f"{[str(path) for path in plans]}"
    )
    extracted = plans[0].parent
    rebuilt = Path(work) / "stage2_resume_input.zip"
    with zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(extracted.rglob("*")):
            assert not path.is_symlink(), "Checkpoint input links are not supported"
            if path.is_file():
                archive.write(path, "results/stage2_test_frozen/" + path.relative_to(extracted).as_posix())
    print("Kaggle unpacked the input ZIP; recovered checkpoint from:", extracted)
    return str(rebuilt)


if RESUME_BUNDLE == "auto":
    RESUME_BUNDLE = find_resume_bundle(Path("/kaggle/input"), WORK)
assert RESUME_BUNDLE and Path(RESUME_BUNDLE).is_file(), "A previous checkpoint ZIP is required"
print("Resume checkpoint:", RESUME_BUNDLE)
'''

SETUP = '''# Record the runtime intervention separately from the unchanged frozen method.
# Explicit placement is supported by Transformers:
# https://huggingface.co/docs/transformers/main_classes/model
assert SYSTEMS == ["no_contract_to_generator"], "This notebook resumes only the unfinished arm"
assert GENERATOR_MODEL != AGENT_MODEL
RECOVERY_VERSION = "gpu-split-sdpa-v3"
checkpoint_counts = {preset: coverage(preset) for preset in ALL_SYSTEMS}
assert all(checkpoint_counts[preset]["complete"] for preset in ALL_SYSTEMS if preset not in SYSTEMS), (
    "The other five systems must already be complete in this checkpoint. "
    "Attach the full previous output bundle."
)
PRESERVED_FILES = {
    str(path.relative_to(OUT)): path.read_bytes()
    for path in OUT.glob("*/*")
    if path.name in ("runs.jsonl", "traces.jsonl", "run_info.json", "prompt_run_info.json")
}
before_keys = {}
for path in OUT.glob("*/runs.jsonl"):
    before_keys[path.parent.name] = [
        [row["task_id"], row["seed"]]
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    ]
completed_before = sum(len(keys) for keys in before_keys.values())
assert completed_before > 0, "No saved runs were restored; do not start a fresh test here"
BOOTSTRAP_PATH = Path(WORK) / "stage2_gpu_split_bootstrap.py"
BOOTSTRAP_PATH.write_text(GPU_SPLIT_BOOTSTRAP)
INTERVENTIONS = OUT / "runtime_interventions"
INTERVENTIONS.mkdir(exist_ok=True)
PLACEMENT_RECORD = INTERVENTIONS / f"gpu_split_resume_{time.time_ns()}.json"
PLACEMENT_RECORD.write_text(json.dumps(dict(
    intervention=RECOVERY_VERSION, reason="CUDA OOM in repair generation on GPU 0",
    scope="new model calls in this resume session only; previous records retained",
    previous_device_map="auto independently for each model",
    model_gpu_ordinals={AGENT_MODEL: 0, GENERATOR_MODEL: 1},
    attention_policy="CUDA EFFICIENT_ATTENTION only; explicitly expand grouped K/V heads; "
                     "no model-inference math fallback; preserve masks, scaling and causal flags",
    source_commit=COMMIT, source_files_edited=False,
    bootstrap_sha256=hashlib.sha256(GPU_SPLIT_BOOTSTRAP.encode()).hexdigest(),
    bootstrap_source=GPU_SPLIT_BOOTSTRAP,
    versions=VERSIONS, checkpoint_sha256=hashlib.sha256(Path(RESUME_BUNDLE).read_bytes()).hexdigest(),
    completed_before=completed_before, existing_task_seed_keys=before_keys,
    selected_systems=SYSTEMS, model_loads=[], attention_events=[],
    limitation="GPU placement and attention backend differ from the earlier run; bitwise output equivalence is not established. "
               "Prompts, model IDs, precision, data, decoding and budgets retain the frozen settings."
), indent=2))
ENV = dict(ENV, STAGE2_PLACEMENT_RECORD=str(PLACEMENT_RECORD))
# Preserve the base command even if this setup cell is run repeatedly.
original_build_command = getattr(build_command, "_stage2_base_command", build_command)


def build_command(preset, _base=original_build_command, _bootstrap=BOOTSTRAP_PATH):
    command = _base(preset)
    return [command[0], str(_bootstrap), *command[1:]]


build_command._stage2_base_command = original_build_command


def verify_preserved_records():
    for relative, original in PRESERVED_FILES.items():
        current = (OUT / relative).read_bytes()
        if relative.endswith(".jsonl"):
            assert current.startswith(original), f"Existing records were changed: {relative}"
        else:
            assert current == original, f"Completed metadata was changed: {relative}"


print("Recovery revision:", RECOVERY_VERSION)
print("Preserved completed records:", completed_before)
print("Pending tasks:", len(TEST_IDS) * len(SEEDS) - coverage("no_contract_to_generator")["records"])
print("Placement: repair 7B four-bit -> GPU 0; generator 3B FP16 -> GPU 1")
print("Attention: fused memory-efficient CUDA kernel; GPU self-test before loading weights")
print("Runtime intervention record:", PLACEMENT_RECORD)
'''


EXECUTE = '''# This cell can also be rerun after the final download/summary cell.
os.chdir(REPO)
failure = None
try:
    # Keep a safe, downloadable checkpoint before any new GPU process starts.
    verify_preserved_records()
    export_bundle("partial")
    for preset in SYSTEMS:
        if coverage(preset)["complete"]:
            print("Already complete; skipping", preset)
            continue
        command = build_command(preset)
        log_path = LOGS / (system_name(preset) + ".log")
        started = time.time()
        print("Running", preset, flush=True)
        with log_path.open("a") as log:
            log.write("Command: " + json.dumps(command) + "\\n")
            proc = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(("[seed", "Verified sandbox:", "Verified GPU placement:", "Attention recovery:", "runs in")):
                        print(line, end="", flush=True)
                code = proc.wait()
            except BaseException:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
            log.write(f"\\nExit: {code}; minutes: {(time.time()-started)/60:.2f}\\n")
        if code:
            print(log_path.read_text()[-7000:])
            raise RuntimeError(f"{preset} failed; inspect {log_path}. Download the partial bundle and resume after diagnosis.")
        assert coverage(preset)["complete"], f"{preset} did not produce all 100 test records"
        verify_preserved_records()
        export_bundle("partial")
        print("Completed", preset, "in", round((time.time()-started)/60, 1), "minutes")
except BaseException as error:
    failure = error
finally:
    try:
        verify_preserved_records()
    except Exception as error:
        print("Preservation check failed; retaining the last safe ZIP:", error)
        if failure is None:
            failure = error
    else:
        try:
            progress = export_bundle("failed" if failure is not None else "partial")
        except Exception as error:
            print("Checkpoint export failed; retaining the last safe ZIP:", error)
            if failure is None:
                failure = error
    from IPython.display import display, FileLink
    try:
        os.chdir(WORK)
        if BUNDLE.is_file():
            display(FileLink(BUNDLE.name))
    finally:
        os.chdir(REPO)
if failure is not None:
    raise failure
'''


def cell(source, kind="code"):
    identifier = hashlib.sha256((kind + source).encode()).hexdigest()[:12]
    out = dict(cell_type=kind, id=identifier, metadata={}, source=source.splitlines(keepends=True))
    if kind == "code":
        out.update(execution_count=None, outputs=[])
    return out


def make_notebook():
    notebook = copy.deepcopy(json.loads((HERE / "stage2_test_frozen_kaggle.ipynb").read_text()))
    cells = notebook["cells"]
    cells[0] = cell(
        "# Resume the interrupted Stage 2 test run\n\n"
        "**Recovery revision: gpu-split-sdpa-v3.**\n\n"
        "Start a **fresh Kaggle notebook session** with **Internet** and **GPU T4 ×2** enabled. "
        "Attach your previous **stage2_test_bundle.zip** as an input. The configuration automatically "
        "finds that filename under `/kaggle/input` and also handles an automatically unpacked ZIP; "
        "if several copies exist, set `RESUME_BUNDLE` "
        "to the exact path. Then use **Save Version → Save & Run All** or Run All.\n\n"
        "Only `no_contract_to_generator` is selected. The runner restores all six systems, "
        "keeps existing task/seed records and skips completed tasks. With the inspected checkpoint, "
        "it starts with **562/600 records**, including **62/100** in the unfinished arm, "
        "and resumes at **taxi_dedup_003** (38 pending). No retraining or GitHub push is needed.\n\n"
        "The previous GPU split was verified but still exhausted memory during attention. "
        "This version keeps the 7B four-bit repair model on GPU 0 and the 3B FP16 generator on GPU 1, "
        "explicitly expands grouped key/value heads, and requires PyTorch's fused memory-efficient "
        "attention kernel during model inference. A small GPU self-test checks causal prefill, "
        "decoding and masks against the math reference on both GPUs **before loading model weights**. "
        "If the installed CUDA build cannot run that kernel, it stops early and exports the checkpoint. "
        "The frozen source checkout, "
        "prompts, parser inputs, precision, decoding and budgets are unchanged. The bundle records "
        "the placement/attention intervention, self-test results, affected session and actual model devices. "
        "Attention kernels can affect numerical results; this recovery is not claimed to be bitwise equivalent "
        "to the earlier execution. The GPU run itself still needs validation.\n\n"
        "Download the new **stage2_test_bundle.zip** when finished. Success requires "
        "`experiment_complete: true` and **600/600 records**.\n", "markdown")
    config = "".join(cells[1]["source"])
    config = config.replace("SYSTEMS = list(ALL_SYSTEMS)", 'SYSTEMS = ["no_contract_to_generator"]')
    config = config.replace('RESUME_BUNDLE = None', 'RESUME_BUNDLE = "auto"')
    config = config.replace("# e.g. first three in session 1, remaining three in session 2",
                            "# Resume this arm; require the other five to be complete")
    cells[1] = cell(config)
    # Restore using the original archive/path/plan checks before installing the wrapper.
    cells[7] = cell(DISCOVER_INPUT + "\n" + "".join(cells[7]["source"]))
    setup = cell("GPU_SPLIT_BOOTSTRAP = r'''" + BOOTSTRAP + "'''\n\n" + SETUP)
    cells[8] = cell(EXECUTE)
    cells.insert(8, setup)
    cells[-1] = cell(
        "## After resuming\n\n"
        "Keep the previous checkpoint and download the newly exported **stage2_test_bundle.zip**, "
        "plus this executed notebook. Confirm **600 records**, **100 per system**, "
        "and `experiment_complete: true`. The resumed bundle includes the earlier OOM log and "
        "a `runtime_interventions/` record documenting the change in GPU placement and attention backend. "
        "Report that intervention with the final experiment; do not describe all records "
        "as having identical device placement or attention kernels. If another error occurs, the failure handler "
        "exports the partial checkpoint again. Do not change prompts, quantization or budgets "
        "to work around a failure.\n", "markdown")
    notebook["metadata"]["igt_recovery_version"] = "gpu-split-sdpa-v3"
    return notebook


if __name__ == "__main__":
    destination = HERE / "stage2_test_resume_kaggle.ipynb"
    destination.write_text(json.dumps(make_notebook(), indent=1) + "\n")
    print(destination)
