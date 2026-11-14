# Tool-use 1.5B training result

**The run completed and selected epoch 2: 90/100 correct-reference passes and 89/100 joint positive/negative successes, versus 14/100 and 10/100 for the unadapted base. The model now uses real input-data tools successfully on all 16 statistic-dependent development tasks.**

There is also a material training artifact: every statistic task first calls `profile_column` on `__unknown_column__`, receives an error, and then calls the correct column. This comes from deliberately erroneous recovery-prefix calls that the training design supervised as assistant targets. The adapter is useful but should not be described as clean or fully efficient tool use.

## Completion and model selection

The full, non-smoke training run completed two epochs and 594 optimizer steps. The delivered adapter is `Qwen/Qwen2.5-1.5B-Instruct` with `parser-tools-v2`, not the selective-facts adapter. Epoch 1 passed the finite development screening rule; epoch 2 scored higher and was selected.

- Base: **14/100** correct references; **22/100** tasks with accepted wrong probes; **10/100** joint successes.
- Epoch 1: **75/100** correct references; **2/100** tasks with accepted wrong probes; **74/100** joint successes.
- Epoch 2: **90/100** correct references; **1/100** task with an accepted wrong probe; **89/100** joint successes.
- Epoch 2 synthetic: **74/80** correct references, one accepted wrong probe.
- Epoch 2 original development: **16/20** correct references, no accepted wrong no-op probes.
- Schema-valid terminal contracts improve from **49/100 to 98/100**. The two remaining terminal failures are invalid-response stops.

Correct-reference and wrong-output acceptance can overlap. The one unsafe datetime contract passes a correct output and a targeted wrong output, so the joint score is 89 rather than 90. These are parser-development scores, not end-to-end Stage 2 accuracy or independent test results.

## What the tools now do

The selected model makes **38 tool attempts across 21 tasks**:

- **16 successful `profile_column` queries** supply the requested median, mean or mode. They use the correct full-column or explicit equality-group scope.
- **16 failed placeholder queries** use `column="__unknown_column__"` first, then recover to the correct query on all 16 statistic cases.
- **Six failed calls** invent unavailable transformation tools: five `rows_removed_where` attempts and one `cast` attempt.

Thus 22/38 attempts fail. This high failure rate is not a broken data tool: the model supplies an invalid placeholder or a nonexistent tool name. Read-only queries and contract predicates/transformation functions are distinct roles, which the adapter still sometimes confuses.

Twenty of the 21 tool-using tasks pass their correct-reference checks, versus zero of 16 tool-using tasks for the base. However, only the 16 successful statistic queries demonstrate useful retrieved information. A failed invented call followed by a correct contract is not successful execution of that invented tool.

All **16/16 statistic-dependent cases** pass the correct and tested wrong-output checks. The base solved 0/16. This is substantive evidence that the trained model can request a needed input fact, read its observation and use the result in a contract for the supported cases. It does not prove general competence with arbitrary tools or natural-language scopes.

## Training artifact: the placeholder prefix

`finetune/tool_use/build_data.py` creates 173 recovery conversations whose first assistant turn intentionally queries `__unknown_column__`. `finetune/tool_use/tokenization.py` supervises every assistant turn, including that deliberately bad initial action. Consequently, the incorrect call receives positive training weight alongside the subsequent recovery.

The exact placeholder is not a column in the evaluation data and is not suggested by the input prompt. Its repeated use as the first action on all 16 statistic cases is consistent with the adapter copying this training pattern. This is an undesirable training-design choice I introduced, not required exploration of the dataset.

The recovery behavior itself works, but the adapter unnecessarily starts with an error. Simply deleting these failed calls from the reported traces would misrepresent the measured behavior. The original data, runtime and adapter remain preserved, including the complete failure traces. No claim is made that the placeholder pattern was the only cause of every failed call, or that its causal effect was isolated by a separate training ablation.

No additional retraining is required to keep or evaluate this adapter. If avoiding further training, possible inference changes—such as explicit valid-column guidance or routing an invalid action back for one correction—need their own version and fresh comparison. They must not silently rewrite requested tool arguments or hide failed attempts. The currently running 3B job should remain reproducible; its source/data have not been changed by this audit.

## The remaining wrong acceptance

`courier_convert_1` requests datetime conversion with **`%d/%m/%Y`**. The final contract checks only `column_dtype: datetime`, row count and other-column preservation. It does not constrain converted values or the date format.

The targeted wrong program parses with **`%m/%d/%Y`**, using `errors="coerce"`. Its output still satisfies the weak dtype-only contract, so the wrong output is accepted. The model first tries a nonexistent `cast` tool; that failure is separate from the weak final contract.

The existing contract vocabulary can express a cast value relation with datetime and a format argument. The current predicted contract omits that constraint. Passing schema validation and a correct reference is therefore insufficient for this task. This is the main reason not to treat the 90/100 number alone as a safety result.

## Remaining correct-reference failures

Ten cases do not accept the correct reference:

- **Four arithmetic edits:** the model creates a new backup-plus column instead of changing the requested existing column.
- **Two other synthetic cases:** one numeric filter remains invalid after repeated unavailable `rows_removed_where` calls; one datetime conversion remains invalid.
- **Adult rename:** it changes `education-num` into nonexistent `education_num`.
- **Adult categorical conversion:** it requires zero missing values despite the request to preserve missingness.
- **Adult remove-any-missing:** it checks only a subset of columns instead of all input columns.
- **Airbnb filter:** it uses `latitude` to test `"Manhattan"` instead of the required location column.

The unsafe courier datetime case is an additional quality failure even though it accepts the correct reference: 10 failed positive checks plus one wrong acceptance gives 11 non-joint successes.

## Compared with selective facts

The corrected selective-facts parser scores **85/100** correct references with zero accepted tested wrong outputs. This tool adapter scores **90/100**, but admits one tested wrong output and costs more model calls/tokens. Both solve the same 16 statistic cases. The tool adapter's five-pass net advantage is not, by itself, proof that tool access is better: the adapters have different training conversations and prompts, and several gains are on non-statistic tasks.

The offline normalization-only selective-facts result is **88/100** with zero accepted tested wrong outputs; it is postprocessing of saved responses rather than the fresh correction evaluation. That correction notebook's GPU result is still a separate pending experiment.

These methods should remain separate candidates until end-to-end comparisons. Never attach this tool adapter to a selective-facts notebook: their protocols and compatibility checks differ.

## Training and inference cost

Training uses 4,743 training conversations and all 914 validation conversations. No sequences were truncated: observed maximum lengths were 3,151 training and 3,072 validation tokens, below the configured 8,192 limit. LoRA has 18,464,768 trainable parameters, rank 16, effective batch 16 and learning rate 0.0002.

Validation loss falls from **0.005038** after epoch 1 to **0.004565** after epoch 2. Final training loss is 0.010130. The recorded training call lasts about **8 hours 22 minutes**, including epoch-callback semantic evaluations; it excludes the earlier baseline inference and final additional validation. Low loss did not prevent placeholder imitation or weak semantic contracts.

Selected-adapter evaluation uses **328,664 tokens and 146 model replies**, with summed parser runtime about **24.7 minutes**. The base used 617,444 tokens and 245 replies. Although the adapter reduces retries/total tokens versus that weak base, it still has 38 tool attempts and the 16 unnecessary placeholder failures. Compared with selective facts, which used 166,045 tokens and 100 replies, the tool protocol is materially more expensive here. Runtime comparisons are diagnostic and exclude loading/setup/CPU scoring.

## Recommendation

Keep this epoch-2 adapter as the trained 1.5B tool candidate. It is a clear improvement over its own base and now demonstrates actual on-demand fact retrieval. Do not start another training run solely on this result. Wait for the ongoing 3B output, and finish the separate selective-facts normalization/correction evaluation.

Before choosing a Stage 2 parser, compare absolute joint success, wrong acceptance, statistic accuracy, unnecessary calls and cost. The `eligible: true` flag only means the declared comparison against this weak base passed; it does not certify the adapter as universally best or safe. Do not erase tool failures or loosen the verifier to obtain a higher score.

## Audit and files

The original uploads are preserved byte-for-byte at:

- `kaggle_outputs/tool_use_v2/tool_use_train_1_5b.zip`.
- `notebooks/executed/tool_use_v2/tool_use_train_1_5b_executed.ipynb`.

Reproduce with `.venv/bin/python tools/analyze_tool_use_training.py`. The audit verifies the complete sealed bundle, actual selected adapter/protocol seals, training source/data hashes, notebook source, task coverage and summary arithmetic. The baseline's identical saved predictions/scores are bound to the prior base audit. Both trained checkpoints receive fresh CPU semantic replay and parser control-flow/tool-observation checks. Tied top-value display ordering is excluded from exact observation comparison; numeric statistics, modes, other observation fields and tool actions are checked. Detailed results are in `audit.json`.

Audit completed: both trained checkpoints' 200 records passed CPU semantic replay and parser control-flow/tool-statistic checks without mismatches. The notebook source matches the published training launcher and has no exception outputs. Source/data, selected adapter and summary checks passed; all 100 baseline final predictions and labels match the previously audited base run. Relative to the scope-fixed selective-facts parser, the selected tool adapter gains eight correct-reference passes and loses three, net +5. All 173 recovery conversations begin with the supervised placeholder call. Original frozen runtime verification also passed.

## Follow-up: normalization and correction checks

Offline inspection of the selected tool adapter's saved final replies found one whitespace-operator failure: `nursery_filter_0`, with `op: "> "`. The same narrow normalizer used for selective facts makes this final contract valid. CPU replay confirms that it accepts the correct transformation and rejects the tested wrong outputs. Postprocessing the saved terminal answers therefore changes the aggregate from 90 correct references / 89 joint successes to **91 / 90**, retaining the one wrong datetime acceptance. This is not a fresh model/tool-loop evaluation; applying normalization earlier in the loop would require its own version and test.

The current tool parser already sends invalid-response correction feedback (up to two correction messages) and returns tool error observations. It recorded ten invalid-response events in this run. Adding another identical generic retry is not a demonstrated solution. Input-only consistency checks additionally identify the bad Adult rename, but do not detect every schema-valid semantic omission, including the unsafe datetime format. Placeholder queries and unavailable tool names remain separate action-policy/training artifacts. The saved runtime and adapter have not been changed. See `normalization_followup.json`.
