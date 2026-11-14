# Existing adapter: normalization and one correction attempt

This is an **inference-only development version**, `p1-v2+sf1-scopefix1-norm1-correct1`. It uses the exact selected epoch-2 1.5B selective-facts adapter and the corrected scope router. No weights are trained, updated, merged or replaced.

## Run on Colab

1. Use your existing Google Drive folder **`MyDrive/igt_scopefix`**, the same folder used by your successfully executed scope-fix notebook.
2. Put these two ZIPs in it (keep the adapter ZIP already there; add the new correction source ZIP):
   - **New source bundle:** `notebooks/kaggle/selective_facts_correction/selective_facts_correction_source.zip`.
   - **Existing training output:** `kaggle_outputs/training/selective_facts_v1/selective_facts_1_5b.zip`.
3. Open **`notebooks/colab/selective_facts_correction/selective_facts_correction_colab.ipynb`** in Colab.
4. Select a GPU with at least 12 GiB memory (a T4 is sufficient for the configured inference workflow), then **Run all**. Authorize the Drive mount. If you chose a different Drive folder, change `DRIVE_FOLDER` in the first code cell.
5. Wait for the three-case inference smoke and then the full 100-case evaluation. Download **`selective_facts_correction_eval.zip`** from the timestamped `results_...` folder in Drive, plus the executed notebook.

Only the new source ZIP and the existing full training-output ZIP are required. You do not need another training run, the original adapter dataset, a previous evaluation result ZIP, a GitHub push, or the tool-training outputs.

## Run on Kaggle instead

Upload the same two ZIPs as private datasets and attach them to **`notebooks/kaggle/selective_facts_correction/selective_facts_correction_kaggle.ipynb`**. Enable Internet and a T4 GPU (T4 x2 also works; only GPU 0 is used). Save Version → Run All. The loader handles both ZIP files and automatically extracted datasets.

Download the full `selective_facts_correction_eval.zip` and executed notebook, not just the smoke output. Place the ZIP under `kaggle_outputs/parsing/selective_facts_correction1/` and the executed notebook under `notebooks/executed/parsing/selective_facts_correction1/`. Include `_colab_executed` or `_kaggle_executed` in its filename.

## The comparison

Every task gets one fresh initial answer using the unchanged scope-fixed prompt. That answer is used by three arms:

1. **Control:** the ordinary scope-fixed parser, equivalent to the method that scored 85/100 previously.
2. **Normalized:** trim surrounding whitespace from recognized operators at schema-defined operator positions, then validate normally. For example, `"> "` becomes `">"`.
3. **Corrected:** start with the normalized answer. If schema or input-consistency diagnostics find a problem, give the same model one feedback message and generate one complete revised contract. The revised answer receives the same narrow normalization and diagnostics.

All three arms use the same initial output, so changes are attributable to the added processing rather than three independent initial generations. There are 100 first model calls and at most 100 additional correction calls. Based on the saved outputs, ten tasks trigger correction, but the fresh run determines the actual count. The three-case smoke is separate from the full run.

The historical 85/100 result is included only as context. The fresh control is the main paired comparison. All 100 cases are included, so a regression on a previously passing task remains visible. Do not add per-task exceptions or select outputs according to the reference score.

## What normalization can change

Only surrounding whitespace on known `op` values at predicate or condition positions is removed. The implementation does **not** trim column names or literal strings, rewrite `<>` as `!=`, invent missing arguments, remove unknown fields, alter tolerances, fix contract versions or weaken preservation predicates. Every normalized field records its path, old value and new value. The original raw answer is preserved.

If there is no eligible operator change, the exact original text is retained. Other schema errors remain errors.

## What triggers the one correction

Diagnostics see only the generated contract and the input DataFrame. They check:

- Contract-schema validation errors, including unsupported fields/functions and missing function arguments.
- Input column references that do not exist. Similar actual names may be suggested, but are never substituted automatically.
- `columns_added` naming a column that already exists.
- Overlapping value-change and value-preservation requirements on rows treated as retained by the contract's explicit retention predicates, or by unchanged row count when there is no declared removal/deduplication predicate.

For value conflicts, the checker evaluates the contract's own relation against input rows and compares it with its own preservation conditions. It does not execute a correct transformation or consult a gold contract. These are bounded diagnostics; they do not establish full intent correctness or detect every contradiction. Unsupported diagnostic computations are not treated as evidence of failure. Conditional deletion and disjoint preservation scopes have dedicated checks against false positives.

The correction prompt contains the original input/request, supplied selective facts, previous contract, and specific diagnostic messages. It asks for a complete correction while retaining all request requirements. No gold contract, reference output, candidate score or task-specific expected answer enters generation or feedback. Evaluation executes the reference/probes **after** the correction decision and generation have finished.

A still-invalid or still-flagged revised contract is recorded as withheld (`correction_failed`), with the revised candidate retained in the trace. It does not fall back to the earlier answer based on a reference score, and there is no third model call. A correction that cannot fit the declared context limit is explicitly withheld. CUDA/infrastructure errors fail the job rather than being disguised as an ordinary parser result.

## Offline result already obtained

On the saved scope-fix outputs, normalization alone recovers the three numeric-filter failures: **85/100 → 88/100**, with zero accepted tested wrong outputs and no changes to the 85 previously passing contracts. The three modified contracts were replayed against their correct and wrong synthetic outputs on CPU.

The diagnostics flag ten remaining saved failures and no previously passing saved contracts. They also flag none of the 80 synthetic gold development contracts. These are useful development checks, not evidence of general zero false positives.

Two saved failures are not detected by this bounded policy: `warehouse_edit_2` invents a schema-valid but wrong operation/columns, and `airbnb_filt_006` selects a real but wrong column. No special-case fixes were added for those task IDs. The correction model may still fail on the ten detected cases; a score above 88 cannot be claimed until inference is run.

See `results/selective_facts_analysis/correction1/offline_audit.json`. Reproduce it with `.venv/bin/python tools/analyze_selective_correction_offline.py`. This is an offline transformation of saved outputs, not a new GPU evaluation.

## Outputs and decision

- `tasks/*.json`: all three candidate contracts and scores; initial and correction messages; both raw replies; normalization paths; diagnostic triggers; rejected revisions; tokens and model-call counts.
- `comparison.json`: cohort scores and paired gains/losses for control → normalized → corrected, plus correction and unresolved task lists.
- `summary.json`: each arm's semantic scores, actual total model calls/tokens and completion status.
- `run_info.json`: source/data hashes, exact adapter seals, actual Python/Torch/CUDA/GPU and library versions, attention checks and inference settings. This environment record is inside the output ZIP for both platforms.
- `completion.json`: `retrained: false` and whether the run was a smoke run.

The per-arm token totals represent each policy's cost; do not add all three arm totals together, since they share first generations. `actual_tokens` records the actual combined job cost. A sealed partial ZIP is refreshed every ten tasks and on handled failures. Existing result folders are refused; Colab copies available ZIPs/logs to Drive on completion or failure. A forced platform termination may prevent the final copy, so no failure-recovery guarantee is claimed.

Choose a method based on correct-reference acceptance, wrong-output acceptance, paired regressions and cost. Better schema validity or a successful correction call is not enough. The verifier is unchanged. These are open-development cases used to design the checks, not independent test accuracy. This package does not modify the Stage 2 workflow or the two running tool-training experiments.

## Implementation

`finetune/selective_facts_correction/` contains the isolated implementation. Rebuild both notebooks and their shared ZIP with:

```
PYTHONPATH=src:. .venv/bin/python -m notebooks.kaggle.selective_facts_correction.make_notebooks
```

Local tests cover exact normalization boundaries, preserving literal data and unknown fields, diagnostics/retained-row scopes, no reference leakage, one-retry bounds, failed correction/context handling, untouched passing cases, real-verifier scripted integration, and both packed/unpacked source layouts on both notebook platforms. Actual GPU correction quality remains to be measured.

Implementation verification: **25 targeted tests passed** (`tests/test_selective_correction.py` and `tests/test_selective_attention.py`). Isolated source ZIP checks, CLI imports, CPU preflight and loading the actual existing epoch-2 adapter passed. Original frozen and scope-fix runtime hashes remain unchanged.

Completed evaluation: **85/100 control → 88/100 normalized → 88/100 corrected**, with zero accepted tested wrong outputs in all three arms. Correction made ten extra model calls without recovering a case; nine revised replies repeated the original exactly. Carry forward the scope-fixed parser with **normalization only**, and disable model correction for the main Stage 2 integration candidate. See `results/selective_facts_analysis/correction1/completed/ANALYSIS.md`. This completed result supersedes the earlier pending-correction status above.
