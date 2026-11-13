# Fine-tuning audit

## Verdict

The saved runs show successful optimization and usable, intact adapters. There is no observed evidence of a crashed run, non-finite saved weights, missing adapter updates, dropped/truncated training examples, or a mismatched parser prompt version explaining the Stage 2 failures. However, the training task and validation design have material weaknesses. These runs should be retained as the original experiment, not treated as a validated solution to semantic contract generation.

The most important problem is **information unavailable in the model input**: some training examples require exact statistics from the entire dataset, while the parser sees only schema, missing counts, and three sample rows. The next priorities are broader semantic validation and better coverage of difficult patterns. Increasing model size or epochs alone does not address these issues.

## What was checked

The read-only [audit script](../tools/audit_finetune.py) produces [machine-readable findings](finetune_audit/audit.json):

```bash
.venv/bin/python tools/audit_finetune.py
```

It checks all 4,410 corrected-data SFT records against their source examples and the current message builder, schema-validates the contract labels, checks train/validation IDs and evaluation-request overlap, recomputes recorded dataset hashes, inspects all adapter tensors, and compares saved run settings with current configurations. The executed notebook and saved loss logs were also inspected. No retraining was performed, and the original data, adapters, prompts, and experiment records were not modified.

This is not a fresh human review of all generated requests and their intended meanings, nor a GPU-level reconstruction of every training gradient.

## Healthy evidence

- Both models trained on **3,614 examples**, monitored a fixed **200-example subset** of the **796 validation examples**, and recorded completion at epoch **2.0**.
- The recorded data hashes match **`finetune/data/v2/sft_train.jsonl` and `sft_val.jsonl`**, confirming use of the corrected data. The notebook copied these files to the default paths on Kaggle, which explains why run metadata names those default paths.
- All **4,410** stored message sequences reconstruct exactly from the corrected source examples; their contract labels validate under the schema. There are no train/validation ID overlaps or exact evaluation-request overlaps, and the existing word-five-gram overlap check finds none.
- Training uses seven source datasets and holds out `student` for validation; evaluation uses Adult, Airbnb, Taxi, and Titanic. Dataset separation is helpful, although shared transformation templates still limit independence.
- Code inspection shows assistant-only loss masking, ignored padding labels, and dropping overlong examples instead of truncating contracts. The saved logs report **zero dropped examples** and a maximum training length of **1,827**, below the configured **3,072** limit. Real tokenizer behavior was exercised by the completed training run; this audit did not retokenize with a GPU model.
- The QLoRA code loads NF4 weights, calls `prepare_model_for_kbit_training`, and attaches LoRA to attention and MLP projections. That preparation follows the [PEFT quantization workflow](https://huggingface.co/docs/peft/developer_guides/quantization).
- The 1.5B adapter contains **392 tensors / 18,464,768 parameters**; the 3B adapter contains **504 tensors / 29,933,568 parameters**. Every tensor is finite and none is entirely zero. Parameter totals match run metadata. Nonzero LoRA B matrices are evidence that adapter updates were saved, not proof that all learned behavior is correct.
- Both adapters record prompt **p1-v2**, and parser metadata records the same version, zero-shot adapter inference, and 120 completed parses per model.
- Recorded gradient norms are finite. Validation loss falls from **0.017676 to 0.013758** for 1.5B. For 3B, it falls from **0.013886 to 0.007926**, then rises slightly to **0.008077**. The small final increase is not sufficient evidence of serious overfitting.

## Issue 1: labels sometimes require hidden information

There are **62 median-fill** and **11 mode-fill** training examples. Their contracts contain literal statistics calculated from the complete source CSV, while their inputs contain only schema, missing counts, and the first three data rows.

For example, one request is "Impute missing Sales using the column median." Its target contract specifies **47.616**. The input shows three Superstore records, with one missing Sales value; it does not supply the full-column median. Across arbitrary datasets, that statistic cannot be reliably recovered from this input.

The labels can be correct for the source data while the prediction task is underdetermined. Repeated schemas and templates may make memorization possible during training, but this does not supply the corresponding statistic for an unfamiliar evaluation dataset. This explains why both the 1.5B and 3B models can guess incorrect Titanic medians despite low token loss.

**Priority:** provide read-only statistic tools or explicitly supplied, correctly scoped statistics. If training tool use, include the tool calls and observations in the training format. More copies of inaccessible literal targets do not solve the information problem. Changing the contract vocabulary to represent aggregate computations would be a separate, explicitly versioned method change.

## Issue 2: the monitored validation subset misses important failure families

The deterministic 200-example sample used by both runs has:

- **0 median-fill examples**, although the complete validation file contains 3.
- **0 mode-fill examples**, although the complete validation file contains 3.
- **0 ordered-category examples**, because the complete validation file has none.
- **0 datetime-conversion examples**, because the complete validation file has none.

The 200 examples are sampled randomly from one held-out dataset, not stratified by difficult pattern. The monitoring loss therefore does not cover these important capabilities. Evaluating all 796 examples would restore the median/mode cases, but would still omit ordered categories and datetime conversion.

**Priority:** introduce multiple held-out validation datasets and deliberately cover each supported pattern. Track free-generation semantic metrics alongside loss. A validation set needs both correct programs that should be accepted and plausible wrong programs that should be rejected.

## Issue 3: difficult training patterns are sparse

The corrected training set contains:

- Ordered categories: **9 examples**.
- Datetime conversion: **13 examples**.
- Most-common-value filling: **11 examples**.
- Text-to-number conversion: **4 examples**.
- Numeric constant filling: **211 examples**.
- Key-based deduplication: **217 examples**.

These are example counts, not equally diverse independent problems. The 3,614 training messages correspond to **1,863 unique dataset/reference-program/contract combinations**; multiple request wordings and some duplicates account for the remainder. Paraphrases are useful augmentation, but do not establish broad compositional coverage.

**Priority:** add reviewed examples of ordered mappings, scoped preservation, multi-column conditions, closely named columns, composite deduplication keys, and combinations of edits. Check that each request contains enough information, or that the parser can obtain it through permitted tools.

## Issue 4: low teacher-forced loss is not a semantic evaluation

Most target tokens describe a repeated JSON structure and common preservation rules. A small error in a column, operator, mapping key, or literal can determine correctness while accounting for few tokens. Low average cross-entropy therefore does not establish correct contract meaning.

The test results illustrate this: both fine-tuned models have 56% structural exact match despite low validation losses. Exact match also has limits: it penalizes redundant differences and can fail to distinguish meaningful behavioral consequences. The [Stage 2 audit](STAGE2_FAILURE_ANALYSIS.md) gives stronger evidence by checking actual outputs.

**Priority:** measure parse validity, acceptance of reviewed correct transformations, rejection of reviewed wrong transformations, semantic predicate errors by pattern, and downstream accuracy/coverage/cost. Keep reference outputs outside the model input.

## Issue 5: current defaults do not reproduce the saved runs

The current YAML files specify **one epoch** and the default `finetune/data/sft_*.jsonl` paths. The saved runs used **two epochs**, the corrected `v2` contents, and `limit_val=200`. The current default SFT files have different hashes from the saved-run data.

The training notebook's `SFT_DIR` copy step explains the historical data location; it does not make a current default script invocation equivalent to the completed experiment.

**Priority:** for future runs, use explicit data-version paths, immutable model/tokenizer revisions, resolved configuration snapshots, and complete package versions. Avoid overwriting the original adapters. The base-model revision and bitsandbytes version are not fully pinned/recorded by the existing training workflow.

## Issue 6: optimizer-step provenance needs better logging

Both runs record **226 optimizer steps over two epochs** with 3,614 examples. That implies roughly **32 examples per optimizer update**, rather than the nominal 16 implied by a one-device reading of batch size 1 and accumulation 16. The evaluation throughput also implies roughly two examples per reported evaluation step.

These observations are consistent with a two-device effective batch, but the saved metadata does not record Trainer's resolved GPU topology, world size, or effective batch size. The artifact evidence does not establish the exact mechanism, so this is a reproducibility gap, not a confirmed incorrect-training diagnosis. Effective batch depends on device count as well as per-device batch and accumulation; see [Trainer documentation](https://huggingface.co/docs/transformers/v5.20.2/en/main_classes/trainer).

The current script manually estimates warmup steps from dataset size and the per-device batch/accumulation product. If training uses multiple devices, that estimate can differ from the desired fraction of actual optimizer steps. Record Trainer's resolved total steps and warmup count in future runs rather than assuming the one-device estimate.

## Lower-priority observations

Only final adapters were saved (`save_steps=0`); no intermediate checkpoint can be retrospectively selected. This does not invalidate a fixed-two-epoch experiment, but limits diagnosis and future selection. Choosing a checkpoint on semantic development metrics would be more informative than blindly extending training.

The audit found no simple evidence that the learning rate, LoRA rank, loss masking, or adapter file corruption caused the observed semantic failures. It also cannot prove those choices are optimal. Running more epochs or increasing LoRA rank should follow a controlled development comparison, not be treated as a repair by default.

## Recommended next experiment

Keep these runs as v1. First create a reviewed, pattern-covered contract-development and validation set. Compare the existing 1.5B and 3B parsers on it, add data-grounded statistics, and measure whether correct transformations stop being rejected without admitting more wrong ones. Then decide whether another fine-tuning run is warranted.

If retraining, rebuild targets and inputs together, strengthen the rare patterns, include tool conversations if tools are part of inference, select using semantic validation, and record the exact training topology and data/model revisions. Evaluate the selected method on fresh held-out tasks after freezing it.
