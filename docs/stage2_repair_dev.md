# Run the 3B repair development comparison

The first stage of the [3B improvement plan](stage2_3b_improvement_plan.md) is implemented as a separate opt-in version. It improves repair evidence, requests at most one fresh-code recovery within the existing attempt budget, and adds an independently switchable output-schema guard. Generation/repair model strength is not increased, and no retraining is performed.

## What to upload and run

1. Create one Kaggle dataset from `notebooks/kaggle/stage2_repair_dev/stage2_repair_dev_source.zip`.
2. Import `notebooks/kaggle/stage2_repair_dev/stage2_repair_dev_kaggle.ipynb` as a notebook.
3. Attach the source dataset. No adapter, old result bundle, GitHub checkout or separate collector is required. The notebook discovers the intact source ZIP or Kaggle's extracted files automatically and verifies every packaged file.
4. Enable Internet and a T4 GPU. One T4 is sufficient for the intended model loading; T4 x2 also works. The unquantized FP16 3B model uses one visible GPU, and the attention preflight uses valid local device indices. The notebook checks actual available memory and the loaded model's precision.
5. Run all cells, or Save Version → Run All. Leave the four default ARMS selected for the first comparison.
6. Download `stage2_repair_dev_full.zip` from the printed timestamped run directory, plus the executed notebook. The final cell displays the four-arm summary. If an infrastructure exception occurs, download the log and partial ZIP from that directory; partial or CPU-scripted outputs are not a completed model result.

The notebook is normal Python code cells with Python language metadata. The source ZIP contains no nested result archives, training outputs, adapter weights or prior predictions, avoiding the previous Kaggle directory-name collisions. It is independent of earlier uploaded results, so there is no code push prerequisite.

## Scope: a repair experiment, not the full Stage 2 workflow

There are **17 synthetic development cases**: 12 deliberately flawed starting programs and five already-correct controls. They cover deduplication/record order, complete-record retention, crashed code, categorical assignment, preservation when adding a column, rounding conventions, preserving null rows, unexpected helper columns, and legitimate fill/add/drop/rename/empty-output operations.

The starting programs and fixture contracts are fixed across arms. References are used only by the evaluator. The live model receives the request, input schema, contract, its code and permitted execution evidence; it does not receive reference code, reference output or correctness scores. There is no intent-parser or initial-generator invocation in this notebook. This isolates the repair changes before spending GPU time on another end-to-end Stage 2 run. It does not measure whether the selective-facts or tool parser improved.

The main run has **17 × 4 = 68 task/arm records**. All arms use the same pinned Qwen2.5-3B-Instruct revision as the existing 3B workflow, full FP16 weights, deterministic decoding, no response cache, three proposed fixes and the existing tool/token budgets. Arm execution order rotates across tasks. Every arm starts fresh from the same fixture program; one arm never consumes another arm's repair output.

The four arms are:

- `control`: the original a1-v4 repair loop and verifier.
- `evidence`: compact candidate-specific evidence, short attempt history, tool-output provenance, explicit operator conventions and one bounded fresh-code recovery prompt.
- `schema_guard`: the original repair loop plus an exact output-column-set check based on explicit additions/removals. Renames are represented by declared removal/addition, as in the existing contract vocabulary.
- `combined`: evidence/recovery and the schema guard together.

The evidence arm groups several related changes. Its score alone does not isolate the contribution of examples, history compaction, operator conventions or the recovery prompt; those can be ablated later if the combined evidence arm helps.

## What changed in repair

The old loop already had full failure reports and repeated-code checks. This version materializes a few relevant input/output examples and contract-derived expectations; summarizes earlier attempts; identifies which candidate produced tool observations; and clears unavailable output after a crash. Detailed examples are omitted rather than guessed when row identity is ambiguous.

When a submitted proposal repeats or leaves the same measured failure unchanged, the next prompt can request one fresh implementation. It consumes the existing proposal budget. It is not an extra unbounded retry, and it does not bypass the safety checker, sandbox or verifier. Measurable changes in the observed failure do not automatically trigger the reset.

The schema guard rejects undeclared helper columns, unexpected removals and duplicate output column names. It respects explicitly declared additions and removals, does not change the code itself, and retains all existing verifier failures. It cannot correct a contract that authorizes the wrong transformation.

No task IDs from the old test runs appear in these inference rules. The development fixtures use different schema names and data, though their failure mechanisms were informed by the earlier audit. These are diagnostic development results, not an independent confirmatory benchmark.

## Execution and result checks

The notebook first executes the full suite with scripted fixture repairs on CPU to validate references, sandbox compatibility, all arms, completion and ZIP export. These outputs explicitly carry `model_quality_result=false`; they are not model-performance measurements. It then runs a two-case GPU smoke and the full fresh GPU comparison. CPU success is not evidence that the 3B model can make those repairs.

Only a successful full run has `completion.json` with `complete=true`, the full record count and `model_quality_result=true`. Each ZIP seals its files with SHA-256. The plan records the source/runtime hashes, suite hash, model revision, actual precision, environment, arms and limits. Per-task files include initial/final code, outcome, token usage, actual tool calls, recovery requests, diagnostic time and traces. The final `SUMMARY.md` and `summary.json` report correctness, wrong acceptance, withholding, lost initially correct outputs and cost.

A guard that changes an incorrect acceptance into withholding improves rejection behavior but has not completed that task. Compare both incorrect acceptances and correct completions. Do not select a version solely because it emits more ACCEPTs.

## Local checks and rebuilding

From the repository root:

```bash
.venv/bin/python -m pytest -q tests/test_stage2_repair_dev.py tests/agent/test_repair_agent.py
.venv/bin/python notebooks/kaggle/stage2_repair_dev/make_notebook.py
.venv/bin/python tools/validate_stage2_repair_dev.py
.venv/bin/python -m eval.stage2_repair_dev.runner --out /tmp/repair_dev_new_cpu_check --cpu-check
```

Use a new output directory for each invocation. The runner refuses to mix existing output directories. The checked source ZIP and notebook must be uploaded together after a rebuild because the notebook pins the source manifest hash.

The module `eval.stage2_repair_dev.agent.run_repair` preserves the original repair entry-point arguments and adds an `arm` keyword. The new [combined end-to-end workflow](stage2_combined_3b.md) now calls this module for both parser methods under its own experiment identifier. Earlier completed runners retain their original behavior.

The focused new tests and existing repair-agent regression tests pass: **60 tests**. ZIP and automatically extracted Kaggle inputs are both tested, including rejection of a changed working source file. The original frozen experiment source-integrity check also passes.

Local verification is CPU-only because the workspace has no Torch/CUDA runtime. The completed Kaggle run is analyzed in [the verified GPU repair-development report](../results/stage2_repair_dev_analysis/v1/ANALYSIS.md): control 11/1/5 versus combined 16/0/1 (correct/incorrect accepted/withheld). This is a synthetic repair-only comparison, not full Stage 2 accuracy. See `docs/stage2_repair_dev_validation.json` for the packaged infrastructure check.

## Remaining stages

The contract-correction and information-routing stages of the plan remain separate work: contradictions, categorical column resolution, grouped statistics and the ordinary-word “mean” routing error are not claimed fixed by this repair notebook. The completed four-arm GPU comparison selected combined as the next candidate. Both end-to-end notebooks are now available in [the combined workflow](stage2_combined_3b.md); run them to measure the full-system effect.
