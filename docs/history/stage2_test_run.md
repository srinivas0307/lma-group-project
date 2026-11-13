# Frozen Stage 2 test evaluation

**Current recommended workflow:** use the [six standalone notebooks and CPU collector](../stage2_six_notebooks.md). The older combined evaluation and recovery notebooks below are retained for provenance. The new workflow evaluates each system independently and requires no previous output bundle.

Upload [stage2_test_frozen_kaggle.ipynb](../../archive/notebooks/core/stage2_test_frozen_kaggle.ipynb) to Kaggle. Enable **Internet** and **GPU T4 ×2**, then Run All. The notebook clones the existing development commit `e9aaa31b85225fc14792c3d34b9f24de149f92b1`; uploading adapter weights or pushing this notebook to GitHub is unnecessary.

The default plan evaluates all six systems on all 100 test tasks, with seed 0, greedy decoding, generator Qwen2.5-3B FP16, repair model Qwen2.5-7B in four bits, agent a1-v4, and the original budgets. Gold-contract and proposed systems use generator prompt b2-v2; request-only arms retain b1-v1. Saved parser outputs come from prompted 7B p1-v3 and fine-tuned 1.5B p1-v2. The no-contract-to-generator comparison measures the entire contract-aware prompt, including the b2-v2 instructions, rather than isolating only the presence of contract text.

Before inference, the notebook verifies 218 source/input checksums, exact test coverage in both parser files, selected prompt hashes, and the development inference-package versions. The versioned runner additionally probes pandas/numpy inside the actual sandbox. A changed environment is rejected rather than silently mixed with the frozen experiment.

## Download and resume

The output is `/kaggle/working/stage2_test_bundle.zip`. It is checkpointed after each completed system and on a run failure. A complete result has **600 unique task/seed records**, with 100 per system. The bundle includes logs, traces, per-system metadata, source hashes, the frozen experiment plan, metrics and explicit completeness counts.

To divide the same experiment across sessions, choose a subset of `SYSTEMS` in the configuration cell. Download the partial bundle, attach it as a Kaggle input in the next session, and set `RESUME_BUNDLE` to the attached ZIP path. Select the remaining systems, or leave all six selected to skip completed arms automatically. Keep every method, data, model, package and budget setting unchanged. Current-session outputs are retained when rerunning cells rather than overwritten by an older resume ZIP.

Do not interpret partial metrics as final test results. The bundle's `experiment_complete` flag must be true before reporting the full comparison. Cross-arm generation caching saves compute, so notebook wall times are not a controlled inference-speed comparison.

### Resume the 562-record checkpoint after CUDA OOM

Use [stage2_test_resume_kaggle.ipynb](../../archive/notebooks/core/stage2_test_resume_kaggle.ipynb) in a **fresh** Kaggle session, with Internet and **GPU T4 ×2** enabled. Attach the previous `archive/stage2/test/root_checkpoint/stage2_test_bundle.zip` as a Kaggle input. The notebook automatically finds that filename under `/kaggle/input` and also handles a dataset upload that automatically unpacks the ZIP; set `RESUME_BUNDLE` to its exact path if several copies exist. Use **Save Version → Save & Run All** or Run All. No GitHub push is required.

The inspected checkpoint has five complete systems and 62 completed tasks in `no_contract_to_generator`. The recovery notebook selects only that arm, restores all previous results using the original strict checks, and resumes its remaining 38 tasks, starting with `taxi_dedup_003`. It verifies that previously saved records and completed metadata remain intact.

The original backend independently uses `device_map="auto"` for each model. Its repair generation exhausted GPU 0; the archived logs do not establish the earlier complete placement or GPU 1 usage. The recovery wrapper explicitly places the 7B four-bit repair model on GPU 0 and the 3B FP16 generator on GPU 1. It verifies actual parameter devices and records this intervention, the wrapper source, checkpoint hash, package versions and pre-existing task/seed keys under `runtime_interventions/` in the exported bundle. The frozen source checkout, model IDs, precision, prompts, inputs, decoding and budgets remain unchanged. Placement changes may affect numerical outputs; report this recovery intervention rather than claiming identical execution conditions across all 600 records. The recovery's GPU behavior must still be validated on Kaggle.

Download the new `archive/stage2/test/root_checkpoint/stage2_test_bundle.zip` and executed notebook. Success means `experiment_complete: true`, `completed_records: 600`, and 100 records for each system. Preserve the earlier checkpoint too. If the recovery fails, its failure handler exports another partial checkpoint.

The initial recovery wrapper failed during model loading when `Qwen2ForCausalLM` did not expose `hf_device_map`. The updated notebook tolerates missing or `None` device-map metadata and still verifies actual parameter devices. This logging failure happened before task inference, so the reported 562-record checkpoint still has 38 tasks pending. Re-upload the updated recovery notebook and use the existing checkpoint; prompts and frozen source files do not change.

Recovery revision **`gpu-split-v2`** also fixes repeated setup-cell execution (which previously wrapped its own command recursively) and explicitly starts model subprocesses in the source repository (the download cell changes the working directory). It requires the other five systems to be complete, exports a safe checkpoint before model loading, and retains that ZIP if preservation checks or a subsequent export fail. Cleanup errors are printed without replacing an earlier model failure. The notebook prints its revision in the setup cell and includes readable bootstrap code for inspection. These paths have CPU regression coverage; no claim of a completed GPU run is made.

### Attention recovery after the verified GPU split

The executed `archive/notebooks/executed/stage2_test/stage2_test_resume_uploaded.ipynb` confirmed the v2 placements: repair parameters on GPU 0, generator parameters on GPU 1. Repair still exhausted GPU 0 inside `scaled_dot_product_attention`, requesting 906 MiB with only 488.81 MiB free. It completed no new task: the latest ZIP still contains 562 records and all 22 prior run/trace/metadata files are byte-identical to the original checkpoint. A copy of this failed-run bundle is stored at `archive/stage2/downloaded_outputs/test/gpu_split_v2_failed/stage2_test_bundle.zip`.

The stack establishes failure inside SDPA but does not record the selected kernel or tensor shapes. A likely contributor is native grouped-query attention falling back to the math kernel. [PyTorch 2.11 documents](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html) native CUDA GQA support for FlashAttention and math, and float32 intermediates in the math implementation. This is an inference, not a captured kernel profile.

The current notebook is revision **`gpu-split-sdpa-v3`**. Its process-local adapter explicitly repeats grouped key/value heads using the documented GQA expansion, then requires `EFFICIENT_ATTENTION` for CUDA model inference. It preserves masks, causal flags, scale, decoding, model dtypes and quantization settings. No inference fallback to math is allowed. The implementation and per-session attention events are retained in `runtime_interventions/`; attention kernels can change numerical outputs, so this is a recorded runtime intervention rather than a claim of identical execution across the experiment.

Before loading model weights, a GPU self-test exercises causal prefill, decoding and a masked prefill on **both** T4s, comparing small fused results against the original math reference with `atol=rtol=0.005`. Expect six `kernel_self_test_passed` events before weight loading. If the installed CUDA build cannot run the efficient kernel, the process stops at this early check and exports the checkpoint. Passing the small self-test establishes kernel availability and a small numerical comparison, not that every full repair context fits in memory. Full GPU completion remains unverified locally.

The known verifier gaps remain unchanged. Preserve rejected, escalated and accepted-wrong cases for analysis. Do not tune prompts or verifier rules after seeing test outcomes. This notebook does not retrain models and does not require the adapters: it consumes the saved contracts.

## Local checks

`tests/test_stage2_test_notebook.py` compiles all cells, checks the six-arm routing through the actual runner using one scripted test task per arm, and exercises resume-plan matching, conflict preservation and archive-path rejection. Scripted pipeline checks are labeled invalid as model-quality results. Real GPU test performance is only established after the Kaggle run.
