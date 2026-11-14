# Completed selective-facts normalization and correction evaluation

**Use scope-fixed selective facts with operator normalization as the next Stage 2 integration candidate.** The fresh paired evaluation scores **85/100 control, 88/100 normalized, and 88/100 corrected**, with zero accepted tested wrong outputs in every arm. One correction attempt adds cost without recovering another case.

## Verified results

- **Control:** 68/80 synthetic plus 17/20 original development references pass = **85/100**. Six terminal schema-invalid contracts.
- **Normalized:** 71/80 plus 17/20 = **88/100**. Three schema-invalid contracts. The three numeric filters are recovered, with no lost passes or new accepted wrong probes.
- **Corrected:** 71/80 plus 17/20 = **88/100**. Ten correction attempts, zero additional successes. All ten revised candidates still fail schema/input diagnostics and are explicitly withheld; the summary's ten non-ok outcomes are not all ordinary JSON/schema parse failures.

All 100 fresh initial prompts and raw outputs match the preceding scope-fix run exactly, including across the recorded environment change. All three arms share each task's first answer. Consequently the measured normalization gain is not due to independent initial generations. All 16 statistic-dependent cases remain passing.

These are open-development parser checks, not end-to-end Stage 2 accuracy or independent test results. The tasks were used to diagnose/design the method. The original cohort has only wrong-no-op probes, while the synthetic cohort includes targeted wrong transformations. Zero accepted tested wrong outputs does not guarantee safety against other mistakes.

## What normalization fixed

`warehouse_filter_0`, `courier_filter_0` and `nursery_filter_0` each contain two occurrences of `op: "> "`. The normalizer removes surrounding whitespace only from recognized operators at schema-defined positions, producing `">"`. It does not change literal data, column names or predicate intent. Each normalized contract accepts the correct transformation and rejects its available wrong probes.

Normalization needs **no additional model calls or model tokens**. This fresh result confirms the earlier offline 85-to-88 replay. There is no reason to incur the correction branch's added cost for these three fixes.

## Why correction did not help in this run

The intended ten saved failures trigger diagnostics again. The same existing single-answer selective-facts adapter receives its original request/data/facts, prior answer and specific feedback. The model is actually invoked ten additional times; there is no evidence of a skipped call or a missing feedback message.

- **Nine revised replies are byte-for-byte identical to their initial replies.** They repeat the same unknown-column, missing-argument, existing-column-added or preservation conflict despite feedback.
- **`warehouse_edit_1` changes its answer**, but retains the value-preservation conflict and introduces an unsupported top-level `must_not_change` field. Schema validation rejects it.
- No third model reply is requested. No reference-based fallback or candidate selection occurs. All ten failed revisions are retained in the trace and withheld.

This demonstrates that this particular adapter/prompt/one-retry method does not use the feedback effectively on these cases. It does not prove that all correction techniques or models are ineffective. The adapter was trained to produce a single final contract, so a mismatch with the new correction conversation is a plausible explanation, not a separately established causal result.

The diagnostics did not flag any of the 88 normalized passing cases. There are no measured lost passes versus normalization, but also no gain. Further correction-prompt experiments are optional future development, not a prerequisite for testing the current useful configuration end to end.

## Cost and environment

Control and normalization each require **100 first generations and 166,045 tokens**. The corrected policy requires **110 generations and 185,881 tokens**: **19,836 extra tokens, about 11.9%**, with no additional correct-reference or joint successes.

The job's actual total is 110 model replies, not 310: the arms share their first answers. Per-arm token totals should not be added together. Summed recorded inference time is approximately **27.3 minutes**, excluding setup/model loading, CPU probe execution and smoke.

The result ZIP includes the actual environment record: Python 3.13.16, Torch **2.14.1+cu130**, CUDA 13.0, Tesla T4, one visible GPU; Transformers 5.20.2, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0, pandas 2.3.3 and NumPy 2.1.3. This is the Colab host's Torch build, not the Kaggle training build. Fresh paired arms run in the same environment, and all control predictions match the previous run exactly.

The exact epoch-2 adapter hash is unchanged (`b1adae381d17e95d1eb9abafedcac2d42b1aff57bfe2681b834060901a1826e7`). `completion.json` confirms full completion, non-smoke and `retrained: false`. The executed notebook matches the published Colab launcher and contains no exception outputs.

## Remaining failures and next action

Normalization leaves **12** failed correct-reference checks: four category mappings, four arithmetic edits, one datetime conversion, one Adult rename, one Airbnb location filter and one Airbnb copy-fill. Nine are schema-valid but reference-incompatible, and three are schema-invalid. The correction policy detects ten of these but solves none. The two undetected cases remain the wrong-but-schema-valid `warehouse_edit_2` and `airbnb_filt_006`.

Freeze the tested useful policy as **scope-fixed selective facts plus narrow operator normalization**. Disable the model-correction branch in the main candidate. No additional retraining is needed.

Next implement this precise policy in a separately versioned proposed Stage 2 runner/notebook and run that proposed arm first. Do not simply rerun the original Stage 2 notebooks: they do not yet integrate this corrected/normalized parser policy. Inspect final correct acceptance, incorrect acceptance, withholding and repair cost before spending on the full six-system comparison. Existing repeatedly inspected test tasks remain diagnostic, not fresh confirmatory evidence.

Keep the 1.5B trained tool adapter (90 reference passes, 89 joint successes, one wrong acceptance and repeated placeholder calls) as a separate method. Its larger positive count alone does not make it preferable to the normalized selective-facts candidate (88 joint successes, zero tested wrong acceptances, lower token cost). Compare end-to-end behavior before choosing a final method. The 3B tool-training result is still a separate pending comparison.

## Audit and organized files

Run `.venv/bin/python tools/analyze_selective_correction_run.py` to reproduce the audit. It verifies ZIP seals, full task coverage, current source/input/task hashes, exact adapter seals, notebook source, summary/comparison arithmetic, all 100 policy traces, feedback messages, normalizations, diagnostics, correction decisions and token counters using scripted reply replay.

The 100 identical control predictions are bound to the preceding clean CPU semantic audit. Normalized scores are bound to the existing offline CPU audit of the three modified contracts and unchanged cases. All attempted corrections are explicitly rejected by the replayed input/schema diagnostics, so their withheld outcomes require no speculative reference score. No new GPU calls or redundant sandbox executions are claimed by this audit. There are no policy or score mismatches.

- Output: `kaggle_outputs/parsing/selective_facts_correction1/selective_facts_correction_eval.zip`.
- Executed notebook: `notebooks/executed/parsing/selective_facts_correction1/selective_facts_correction_colab_executed.ipynb`.
- Machine-readable verification: `audit.json` in this folder.

The two uploaded files were moved/renamed for organization without changing their bytes. Training/runtime source and published source bundles were not modified.
