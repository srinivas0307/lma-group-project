# Independent Stage 2 proposed-system analysis

The notebook completed successfully and its bundle passes strict validation. The measured method regressed: **44 correct, 2 incorrect accepted, 54 withheld**, versus **76, 2, 22** in the earlier frozen proposed run. This is a valid completed run, not an improvement.

## Runtime and input validation

- All 100 task records and 100 fresh contracts are present; the ZIP and file hashes validate.
- Notebook execution errors: 0; source matches the current template: True.
- Exact reconstructed parser messages that differ from saved inputs: 0.
- CPU replay label mismatches: 0; reference-verdict mismatches: 0.
- The torchao setup fix succeeded. The dependency-conflict warnings did not stop the job.

## What happened

The parser produced 91 schema-valid contracts, seven schema-invalid contracts and two invalid JSON outputs. Of the 91 valid contracts, **46 accept the correct reference output and 45 reject it**. The earlier fine-tuned contracts gave 81 PASS, 16 VIOLATION, one UNCERTAIN and two parsing failures.

The 54 withheld tasks comprise **nine parse failures, 43 tasks whose contracts reject the correct reference output, and two tasks whose reference output passes but whose search/repair does not deliver a correct accepted result**. The other two reference-rejecting tasks are the two incorrect acceptances.

Generation initially produced **71 correct outputs**. The system withheld **32** of these and repaired **five** initially wrong outputs: **71 − 32 + 5 = 44**. The audit also found 8 withheld tasks with a correct later submitted repair: adult_col_005, adult_dedup_003, airbnb_conv_003, airbnb_conv_005, airbnb_dedup_005, airbnb_filt_001, airbnb_filt_005, titanic_dedup_004. These observed candidates do not imply a deployable selector could recover them without better checks.

## Direct examples

- `titanic_miss_004`: the input explicitly supplies **Age median 28** and **mean 29.69911764705882**. The parser chooses the mean, applies it without a missing-row condition, demands that nulls increase, and preserves the originally missing cells. The generator nevertheless produces a correct median-fill implementation; the wrong contract rejects it. Information was available but was misinterpreted.
- `titanic_miss_002`: the parser correctly selects mode **S**, but requires every Embarked value to become S instead of only filling missing cells. Correct literal selection alone does not fix scope.
- `taxi_dedup_001`: the generated code correctly deduplicates the requested composite key, while the contract uses `target: "*"`, so it rejects the correct output.
- `taxi_miss_002`: the request says to mark RatecodeID 99 as missing; the contract instead requires clipping to [-1,99]. The generated clip passes the wrong contract and is incorrectly accepted.
- `taxi_miss_005`: the request says to mark passenger_count 0 and RatecodeID 99 as missing; the contract instead clips both columns to [-1,0], and that incorrect transformation is accepted.

## Important configuration confound

The original saved parser metadata records **NF4 4-bit inference** (`load_in_4bit: true`), Torch 2.11.0+cu128, Transformers 5.16.1, and batch four. This independent run uses an **FP16, non-quantized parser**, Torch 2.11.0+cu130, Transformers 5.20.2, batch one, and the new profile input. The adapter hashes are recorded and match the expected model artifact.

The new implementation changed the parser loading configuration as well as its inputs. Thus this result cannot establish whether the richer profile, quantization change, library changes or their interaction caused the regression. A QLoRA adapter can be loaded over an unquantized base; this difference is a material experimental confound, not proof that FP16 alone is the cause. The current completed version and its collector should stay unchanged for reproducibility.

## Comparison and cost

There are 3 newly correct tasks and 35 lost correct tasks relative to the old proposed system. Incorrect acceptances remain two but occur on different tasks. Per-family and per-dataset counts are in `audit.json`.

This run records **1,038,787 total tokens**, including **335,564 parser tokens**. The earlier reported Stage 2 token cost excluded parser inference. Adding the saved test-parser calls back gives **673,401 old total tokens** (504,239 generation/repair + 169,162 parser), versus the new total above. These are inference token counts, not monetary or wall-clock cost estimates. Summed recorded parser/runtime duration is 44.9 minutes; it excludes setup, model loading, reference evaluation and the smoke run.

## Recommended continuation

1. Preserve this result as the completed FP16/profile diagnostic experiment; do not overwrite or call it a successful quality improvement.
2. Before spending GPU time on the other five systems, run a small controlled parser comparison on separate development data: original input versus enriched input, crossed with NF4 versus FP16. Hold adapter, base revision, package versions, decoding and tasks fixed. That isolates the loading and information effects.
3. Address scope, contradictory preservation and wrong-key/operator errors in semantic development tests. Merely supplying more statistics did not resolve them in this run.
4. The separately implemented tool version is still exploratory: it also currently uses FP16 for the fine-tuned parser, and the adapter was not trained for tool conversations. Resolve the loading comparison before interpreting a later tool/profile difference as a tool benefit.
5. After choosing an input protocol, build matching training examples and semantic checkpoint validation. Treat old-test improvements as diagnostic; reserve a reviewed independent benchmark for confirmation.

CPU replay uses Python 3.12.3 and pandas 2.3.3; the submitted job used Python 3.13.16. The machine-readable audit reports any observed cross-environment differences. No GPU model calls, adapter changes, runtime edits or result rewrites were made during this analysis.

Reproduce with `.venv/bin/python tools/analyze_stage2_independent_proposed.py`. Replay scratch caches are kept under the system temporary directory; the complete case evidence is saved in `audit.json`.
