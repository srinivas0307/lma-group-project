# Stage 2 v2: parser loading correction and development diagnosis

**Diagnostic completed:** original/NF4 passed 16/20 references versus profile/NF4 8/20; original/FP16 15/20 versus profile/FP16 7/20. NF4 did not resolve the profile regression. See the [validated diagnostic analysis](../results/stage2_independent_analysis/parser_diagnostic_v1/ANALYSIS.md). Defer the six full profile runs while testing a smaller information format on broader development coverage. The instructions below document the reproducible diagnostic and full-run artifacts.

The submitted v1 notebook had no runtime exception. Its completed result was 44 correct, 2 incorrect accepted and 54 withheld. Many generated contracts rejected correct transformations. Supplying data did not ensure the parser used the right statistic, row scope, key or operator.

One implementation mistake was confirmed: v1 changed the original fine-tuned parser's NF4 inference to FP16. Version 2 restores NF4 (FP16 computation, no double quantization, matching the original inference configuration) and pins the base model to `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, the revision recorded in the submitted v1 run. This controls revision drift between these new comparisons; the historical original base revision is unknown. Generator, repair, adapter weights, decoding and profile/prompt format remain as in independent v1. The newer library environment remains; this is not an exact reproduction of the historical parser.

**Quality recovery is not yet verified.** NF4 is a configuration correction, not proof of the cause of the regression. No reference-based rewriting, relaxed verification or gold-derived repair is introduced. Scope errors may still require matching training examples and semantic validation.

## Run this notebook first

1. Upload `notebooks/kaggle/stage2_independent_v2/stage2_parser_diagnostic.ipynb` to Kaggle.
2. Select **GPU T4 ×2**, enable **Internet**, and attach your original 1.5B adapter dataset at `/kaggle/input/datasets/srinivasu2005/qwen-1-5b-adapter`. Automatic discovery supports files directly in this folder. No 3B adapter or previous output is needed.
3. Use a fresh session, then **Save Version → Save & Run All**.
4. Wait for **COMPLETE: four controlled development arms**.
5. Download `stage2_parser_diagnostic.zip` and the executed notebook. Put the ZIP in `kaggle_outputs/stage2/independent_v2/` and the executed notebook in `notebooks/executed/stage2/independent_v2/`.

The notebook generates contracts for all **20 existing development tasks**, independently in four arms: original input + NF4; profile input + NF4; original input + FP16; profile input + FP16. That is 80 parser calls, not 80 independent tasks. Each arm loads the same pinned base and adapter in its own subprocess. It needs no generator or repair model, and does not train anything.

Each output saves the exact prompts/replies, schema-validation result, reference-output verdict, wrong-no-op probe, tokens, package versions and source/input hashes. The summary includes paired gained/lost reference passes. Scoring runs after inference; model input contains only the request and raw-data information. A reference pass does not establish complete contract correctness. The wrong-no-op probe checks only one error type; inspect failures and expand development coverage before choosing a final method. The 20-task comparison is an initial diagnosis, not a strong research claim or automatic model-selection gate.

## Updated full workflow

`notebooks/kaggle/stage2_independent_v2/stage2_independent_v2_notebooks.zip` contains the diagnostic plus six independent full-run notebooks and a collector. After reviewing the diagnostic, the updated proposed-system notebook is `stage2_05_proposed.ipynb` in that same v2 folder. Full runs still compute fresh contracts, generation, verification and repair in one job. Diagnostics are recommended checks, not runtime inputs or required files.

Full result names are `stage2_independent_v2_<system>.zip`; collect only these with `stage2_independent_v2_collect.ipynb`. Do not mix v1, v2 or tool-version bundles. The v2 collector rejects the old result. All notebooks embed source/raw data, so no GitHub push is required.

The completed v1 code, templates and result are preserved. The separate tool version remains unchanged and exploratory. Its FP16 setting should not be interpreted as a controlled tool-vs-profile comparison until loading is matched in a future version.

## Local validation

Run `.venv/bin/python -m pytest tests/test_stage2_independent_v2.py -q`.
Regenerate templates with `.venv/bin/python notebooks/kaggle/stage2_independent_v2/make_notebooks.py`.
CPU checks exercise all six routes, parser isolation, strict bundle validation, embedded source bytes, public-only diagnostic messages and wrong-no-op detection. They do not measure GPU model quality or guarantee Kaggle execution.
