# Selective-facts 1.5B training result

The supplied `stage2-selective.ipynb` is the selective-facts **training** notebook despite its filename. The full job completed, selected epoch 2, and exported a real non-smoke adapter. It did not run end-to-end Stage 2.

## Main results

- Original adapter with its original input: **68/100** correct-reference acceptances (52/80 synthetic, 16/20 original development tasks).
- New adapter, epoch 1: **62/100** (47/80 synthetic, 15/20 original). It failed the screening rule.
- New adapter, epoch 2: **73/100** (56/80 synthetic, 17/20 original). It passed the implemented screening rule and is the delivered adapter.
- Tasks with an accepted wrong-output probe: **4 → 0**. This checks a finite set of wrong transformations, not all possible mistakes.
- Original-to-epoch-2 paired changes: **12 gains, 7 losses**, net +5. Synthetic: 11 gains, 7 losses. Original development: 1 gain, 0 losses.
- Epoch 2 withholds a correct reference output on **27 tasks**: 12 blocked by facts routing, 6 invalid model contracts, and 9 schema-valid contracts that fail semantic verification. These are parser-level reference checks, not 27 failed end-to-end repairs.
- Total recorded parser tokens across the 100 tasks: 153,102 original versus 145,496 epoch 2. The 12 routing failures make no model call, so this is not a controlled efficiency improvement.

## A material implementation bug

**Twelve of the 16 synthetic statistic-dependent tasks never reached the new model.** All full-column median, mean and mode tasks include the sentence:

> Ignore missing values when computing statistics.

The SF1 router treats the word `when` as evidence of conditional scope. It then demands an explicit `column = value` group, even though these requests ask for full-column statistics. It returns `FactUnavailable: Conditional statistic needs an explicit column = value scope` before constructing the prompt or generating an answer.

This is a bug in the implemented scope router. These twelve failures are not evidence that the adapter could not use correct supplied facts. The same bug appears at both training checkpoints. The original-adapter baseline does not use this router, so the comparison includes different routing failure modes.

The four explicitly grouped median tasks **do** receive facts: the original adapter passes 0/4; epoch 2 passes **4/4**. This is encouraging evidence for the combined selective-facts-plus-training approach on this small group. It does not isolate the effect of facts from retraining, and does not establish full-column mean/median/mode performance.

The audit removes only the neutral “Ignore missing values when computing statistics.” sentence to reproduce the routing diagnosis on CPU. This is a diagnostic counterfactual: it does not change the experiment, run the model again, or create replacement scores. Corrected facts delivery still needs a new inference evaluation.

## What improved and what still fails

Epoch 2 fixes all four conditional-doubling tasks. The original adapter accepted targeted wrong transformations on these tasks; the new contracts reject those tested wrong outputs and accept the correct ones. It also fixes all four grouped median cases, three datetime conversions, and the Titanic development filter.

Remaining adapter errors include:

- **Four category mappings:** contracts request the mapping but also preserve the original nonmissing values, contradicting the requested change. These four cases were passed by the original adapter and are regressions.
- **Three numeric filters:** an operator is emitted as `"> "` with a trailing space, so schema validation rejects it. These are three further regressions.
- **Add-to-existing-column tasks:** some contracts say an already existing column must be newly added; others invent columns such as `add`. Two correct outputs receive UNCERTAIN rather than PASS because the contract contradicts the input schema.
- **One datetime conversion:** an unsupported `format` field makes the contract invalid.
- **Adult rename:** the contract changes the input column name `education-num` into `education_num`.
- **Airbnb filter:** the contract uses `neighbourhood` where the request requires `neighbourhood_group`.
- **Airbnb copy-fill:** the `identity` relation omits its source argument and fails validation.

A small loss or an overall improvement does not remove these individual regressions. The screening rule compares aggregate cohorts, not every task or transformation family. Zero accepted wrong probes can also result from rejecting correct and wrong outputs together; it does not by itself establish useful correctness.

## Training health and notebook output

The full run trained 4,003 examples and evaluated 902 validation examples, with no sequence truncation (longest observed training sequence 1,976 tokens; validation 2,080, below the 4,096 limit). It completed 502 optimizer steps over two epochs, using LoRA rank 16, learning rate 0.0002 and effective batch 16 on a T4.

Validation loss decreased from **0.006895** at epoch 1 to **0.006326** at epoch 2. Final training loss was 0.009872. The recorded training call lasted about **5 hours 24 minutes**, including semantic evaluations performed inside epoch callbacks; it excludes the earlier baseline evaluation and the final extra validation pass. Loss is useful for optimization health but does not replace semantic scores.

The notebook has no exception outputs. The package resolver printed conflicts with other preinstalled Kaggle packages; they did not stop this job. CPU Trainer/LoRA and both masked-GPU attention forward/backward preflights passed. The smoke adapter failing quality screening is expected for the tiny infrastructure run and does not invalidate the completed full job.

## Recommendation

**Keep the epoch-2 adapter, but do not spend on all six full Stage 2 runs yet.** The first follow-up should fix the facts router in a separately versioned protocol and rerun parser inference with these existing weights on the same development cases. No immediate retraining is needed to determine whether delivering the missing facts fixes the 12 blocked cases.

Preserve the current bundle and metadata. SF1 binds the adapter to the facts implementation hash, so silently editing `context.py` would make the current adapter fail compatibility validation and would break experiment reproducibility. A corrected evaluation must explicitly record the new facts protocol and the unchanged adapter weights.

After that parser check, run the proposed Stage 2 arm first using a compatible, separately versioned workflow, then the full comparison if warranted. Further training should target the observed scope, mapping/preservation, exact-name and schema-format errors. Do not claim these development results as independent test accuracy or claim that epoch 2 is universally the best adapter.

The two new tool-use workflows remain separate: this adapter was trained to consume automatically supplied selective facts, not to issue tool calls.

## Reproduce the audit

Run `.venv/bin/python tools/analyze_selective_facts.py` from the project root with the two supplied files present. `audit.json` records ZIP/adapter/source checks, reconstructed prompts, CPU replay, paired cases, pattern counts and the routing counterfactual. The audit makes no GPU calls and does not modify training or inference code.

Audit completed: all 300 task records (100 tasks across three stages) were checked. There were zero reconstructed-prompt, extraction, CPU-score or summary mismatches. All 66 recorded runtime source hashes match, the original control adapter seals match, the delivered adapter passes compatibility validation, and the notebook source matches the published template. Removing the neutral sentence lets the existing router compute one fact in each of the 12 blocked cases; no new model inference was performed.

Follow-up implementation is now available: see `docs/selective_facts_scopefix.md` and `notebooks/kaggle/selective_facts_scopefix/selective_facts_scopefix_eval.ipynb`. It evaluates the same sealed epoch-2 weights with the separately versioned `p1-v2+sf1-scopefix1` router. CPU preflight verifies facts delivery for the 12 blocked cases and exactly unchanged prompts for the remaining 88; fresh GPU scores are still pending.
