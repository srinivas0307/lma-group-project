# Stage 2 with unquantized 3B generation and repair

This is the current requested configuration: **Qwen2.5-3B-Instruct for both initial code generation and the repair loop, with full FP16 weights and no 4-bit/8-bit quantization for either role.** The 7B model is not loaded by this version.

Both roles use the exact same pinned model revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. They have different generation/repair prompts, but neither receives a stronger base model. The existing selected **1.5B intent-parser adapters** retain their previously tested NF4 setup. No model is retrained and adapter files are not rewritten.

The experiment is `stage2-dual-parser-3b-fp16-v1`, implemented in `eval/stage2_dual_3b/`. Outputs have distinct `stage2_dual_3b_...` names, and the collector rejects the older 7B-repair experiment or a quantized repair configuration.

## Kaggle: recommended single-upload packages

For selective facts:

1. Create a private Kaggle dataset from **`kaggle_uploads/stage2_dual_3b/stage2_dual_3b_selective_kaggle_dataset.zip`**.
2. Import **`notebooks/kaggle/stage2_dual_3b/stage2_dual_3b_selective_proposed_kaggle.ipynb`** and attach that one dataset.
3. Enable Internet and T4 GPU (T4 x2 also works), then Save Version → Run All.
4. Download **`stage2_dual_3b_selective_proposed.zip`** and the executed notebook.

For tool use:

1. Create a separate private Kaggle dataset from **`kaggle_uploads/stage2_dual_3b/stage2_dual_3b_tools_kaggle_dataset.zip`**.
2. Import **`notebooks/kaggle/stage2_dual_3b/stage2_dual_3b_tools_proposed_kaggle.ipynb`** and attach that one dataset.
3. Run all with Internet and a GPU, then download **`stage2_dual_3b_tools_proposed.zip`** and the executed notebook.

Each clean package contains the matching source and its exact selected adapter. They contain **no nested ZIPs, baseline result folders or semantic checkpoint archives**, avoiding the earlier Kaggle extraction collisions. Do not upload the full tool-training result ZIP as the Kaggle dataset. Automatically extracted inputs are supported.

Use separate sessions for the two methods. They can run independently or in parallel if your available quota permits. They do not require each other's outputs, previous parser scores, Stage 1 results or old generated code.

For comparison:

1. Create **separate datasets** for the two full output ZIPs, because their internal result-folder names overlap.
2. Import **`notebooks/kaggle/stage2_dual_3b/stage2_dual_3b_collect_kaggle.ipynb`** on CPU.
3. Attach both result datasets plus either of the matching clean input datasets, which already contains the source. Alternatively attach `notebooks/kaggle/stage2_dual_3b/stage2_dual_3b_source.zip` as the source dataset.
4. Run all and download **`stage2_dual_3b_comparison.zip`**.

The collector accepts intact result ZIPs or automatically unpacked result datasets and checks the original sealed contents. Keep Python/package versions matched across the two GPU runs; differing environments or old 7B-repair bundles are refused.

## Colab alternative

Use your existing **`MyDrive/igt_scopefix`** folder:

- Add the new **`notebooks/kaggle/stage2_dual_3b/stage2_dual_3b_source.zip`**.
- For selective facts, keep the original **`selective_facts_1_5b.zip`** training output there and run **`notebooks/colab/stage2_dual_3b/stage2_dual_3b_selective_proposed_colab.ipynb`**.
- For tools, keep **`tool_use_train_1_5b.zip`** there and run **`notebooks/colab/stage2_dual_3b/stage2_dual_3b_tools_proposed_colab.ipynb`** in a separate session.
- Enable a T4 GPU or another CUDA GPU with at least 12 GiB memory. Run all.
- Results are copied to timestamped Drive folders. After both finish, run **`notebooks/colab/stage2_dual_3b/stage2_dual_3b_collect_colab.ipynb`** on CPU using the new source ZIP and both full result ZIPs under the same Drive folder.

The clean combined packages are for Kaggle's automatic extraction. For Colab's file-based loader, use the new source ZIP plus the original adapter/training-output ZIP, as above. Do not use the older `stage2_dual_source.zip` in these new notebooks; its source/model configuration is different.

## What the two methods do

**Selective facts:** the existing epoch-2 1.5B selective-facts adapter, corrected scope router and narrow recognized-operator whitespace normalization. One initial parser answer; the model-correction branch is disabled, as the completed comparison found no benefit from it.

**Tools:** the existing epoch-2 1.5B tool adapter, bounded read-only tool loop and the same narrow normalization on final-contract replies. Tool failures and placeholder queries remain recorded. No hidden replacement of columns or tool arguments is introduced.

Both then use **unquantized 3B FP16** code generation → execution/verification → **unquantized 3B FP16** repair as needed → reference evaluation. Runtime checks reject quantized or non-FP16 generator/repair floating parameters. The backend never configures bitsandbytes quantization for those two roles; only the intent parser uses NF4.

The repair architecture and limits are preserved: `a1-v4` repair prompt; at most three repair proposals, eight repair tool calls per attempt and 60,000 repair tokens; 512 maximum initial-generation tokens and 768 per repair reply. The proposed generator uses the existing contract-aware `b2-v2` prompt. The verifier, sandbox, facts router, parser adapters and tool budgets are otherwise inherited from the preceding two-parser pipeline.

A two-task infrastructure smoke runs first, followed by a fresh full 100-task run. ACCEPT, REJECT or ESCALATE can be legitimate smoke outcomes; a completed smoke does not guarantee high model quality.

## Loading and resource behavior

Only one model is resident at a time. The runner parses all tasks, unloads the parser, generates initial code with 3B FP16, unloads it, and loads the same 3B FP16 weights for repair. The model download is shared through the normal model cache; weights do not need to be downloaded twice. On one GPU all roles use GPU 0; with two GPUs generation uses GPU 1 and repair uses GPU 0, while remaining sequential.

The repair phase consumes the exact one-use initial response freshly generated in that same run, with matching-prompt assertions. It does not call the generator twice or reuse a prior notebook's predictions. Costs include the parser, initial generation and repair. Model loading/setup is outside the per-task timing totals. This reduces simultaneous model residency but does not guarantee against quota limits, interrupted runtimes or every GPU memory failure.

The repair model is now the same size, precision and revision as the code generator. To assess the architectural benefit in a research claim, compare against direct 3B and appropriate 3B retry/checking baselines with clearly reported budgets. Using the same model removes the stronger-repair-model explanation, but architecture gains still need measured results rather than an assumption.

## Versioning and collection

The original 7B-repair workflow, source ZIPs, notebooks and old results remain unchanged for provenance. **Use only the new `stage2_dual_3b_...` notebooks, source/input packages and collector for this experiment.** If you already started an older run, it cannot be converted to the new experiment or mixed into the new collector; a fresh 3B-repair run is required.

The collector verifies experiment/model/precision settings, exact selected adapter seals, pinned revisions, matching source manifests and software environments, complete fresh task/contract/generation records, tool failures and token accounting. It reports correct accepted transformations, incorrect accepted transformations, withholding, initially correct outputs lost, repairs gained, parser/reference incompatibility and cost.

The underlying six system presets remain available in the new runner and same-method six-system collector. This delivery starts with the two proposed systems and paired collector; it does not require all baseline combinations upfront. All presets now use the declared 3B FP16 agent role, including a prompted-parser baseline when selected.

The test tasks have been repeatedly inspected during development. Treat these runs as diagnostic end-to-end comparisons; independent benchmark evidence is still needed for stronger generalization claims.

## Save and share outputs

Place full result ZIPs under **`kaggle_outputs/stage2/dual_parser_3b_fp16_v1/`** and executed notebooks under **`notebooks/executed/stage2/dual_parser_3b_fp16_v1/`**. Keep repeated runs in separate subfolders.

Share the two proposed result ZIPs, the collector ZIP and executed notebooks after completion. Partial ZIPs/logs are exported on handled failures and periodically during the run; no automatic-resume or forced-platform-kill recovery guarantee is claimed.

## Rebuild and validation

- Notebooks/source: `PYTHONPATH=src:. .venv/bin/python -m notebooks.kaggle.stage2_dual_3b.make_notebooks`.
- Clean Kaggle packages: `.venv/bin/python kaggle_uploads/stage2_dual_3b/make_kaggle_datasets.py`.
- CPU preflight: `PYTHONPATH=src:. .venv/bin/python -m eval.stage2_dual_3b.preflight`.

Validation covers both normalized parser paths, sequential model lifecycle, real sandbox repair, six retained system routes, context refusal, collector rejection/mixing, packed/unpacked notebook inputs, exact same 3B generator/repair configuration, unquantized FP16 model-load arguments and rejection of quantized/wrong-dtype repair weights. Both clean packages are tested with the actual selected adapter files. No local GPU/model-quality result is claimed.

Completed local verification: **32 tests passed** across `tests/test_stage2_dual_3b.py` and `tests/test_selective_attention.py`. Both clean Kaggle packages passed the current notebook loaders, isolated CLI/preflight checks, exact runtime-hash comparison and validation of the real selected adapters. All six new notebook files validate/compile. Model-load tests explicitly verify no quantization configuration for generator/repair and rejection of quantized/FP32 weights. The collector checks recorded loaded precision and refuses 7B or quantized repair settings. Full GPU quality is pending; the original 7B source pack and frozen project sources remain unchanged.

The first full runs are complete and audited: both methods score **73 correct / two incorrect accepted / 25 withheld**. Tools has no net quality gain and uses 17.7% more total tokens; selective facts is the current lower-cost candidate. See `results/stage2_dual_3b_analysis/ANALYSIS.md` for the shared wrong acceptances, withheld-task decomposition, router false positive and matching-baseline recommendations. This completed audit supersedes the earlier pending-GPU-quality status for these two supplied runs; it does not change the frozen runtime or claim independent benchmark performance.
