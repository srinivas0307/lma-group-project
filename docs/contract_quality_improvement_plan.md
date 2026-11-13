# Contract-quality improvement plan

Status: initial synthetic draft suite, CPU semantic harness and two existing-adapter pilot notebooks and a CPU collector implemented. The default-profile and semantic-checking tracks now run together as `cq-profile-check-v2`, with exact input capture, prompt preflight, provenance validation and readable rejection explanations. See the [pilot runbook](contract_quality/RUNBOOK.md). All 120 supported draft tasks pass automated checks. Independent review, coverage expansion, GPU pilots and retraining remain pending; no new model-quality results are claimed.

The later integration is now available as the [independent Stage 2 workflow](stage2_independent.md): it generates contracts from raw data inside each job, includes parser cost, and needs no earlier evaluation outputs. Its GPU runs remain pending. The parser pilot is optional rather than a prerequisite for this workflow.

## Objective and evidence

Improve correct task completion while retaining the proposed system's low incorrect-acceptance rate. The current result is 76 correct accepted outputs, 2 incorrect acceptances, and 22 withheld tasks. Direct generation produces 78, 20, and 2 respectively.

The [failure analysis](../results/STAGE2_FAILURE_ANALYSIS.md) identifies contract/parsing incompatibilities on 19 of the proposed system's 24 unsuccessful tasks; five further failures occur despite reference-compatible contracts. The [fine-tuning audit](../results/FINETUNE_AUDIT.md) finds inaccessible statistic targets, missing validation patterns, sparse difficult examples, and incomplete training provenance. Thus contract quality is the first priority, but improving it does not guarantee that all generation/repair failures disappear.

Recommended sequence: **semantic development suite → existing-model comparison → information-grounding pilot → revised training data → one controlled fine-tuning run → locked selection and pipeline check → independent evaluation**.

## 1. Preserve v1 and define the new experiment

Keep existing results, adapters, prompt versions, data, and frozen input hashes as v1. Assign new identifiers to data, prompts, model adapters, and output directories. Record that v2 design was informed by the old test failure analysis; improvements on those old test cases are diagnostic only.

Start with Qwen2.5-1.5B and the already trained 3B adapter. Use prompted 7B as a reference if needed. A new 7B training run is not the first experiment: the existing 3B and prompted 7B results do not show that size alone solves the problem.

Deliverable: a versioned experiment specification listing the input information, tools, candidate models, budget, metrics, and selection rule before model execution.

## 2. Build reviewed semantic development and selection data

Initial target: 120 new tasks, with 80 open development tasks and 40 locked selection tasks. These are development/selection resources, not the eventual independent test set. Forty tasks are a practical screening set, not enough to certify generalization or reliably resolve small differences.

Use several new source datasets and schemas absent from the original fine-tuning and evaluation datasets. Separate open development and selection by source dataset where feasible, and prevent near-duplicate requests or template instances across splits. Keep the five transformation types represented in both splits.

Deliberately cover:

- Full-column and conditional/grouped statistics, including missing data and mode ties.
- Ordered categories, numeric mappings, and datetime conversion with explicit format.
- Similar column names and ambiguous-looking category/key choices.
- Single and composite deduplication keys, keep-first/keep-last, and keep-none semantics.
- Correctly scoped preservation and multiple columns with different row conditions.
- Multi-edit requests, comparisons at boundaries, and unsupported/ambiguous requests.

For each supported task, write the request, assumptions, input data, gold contract, reference program, at least one alternate correct implementation where feasible, and a small set of targeted wrong programs. Include cases where a nearly correct contract should fail to accept the wrong program. Unsupported tasks need an explicit expected escalation label and a separate outcome metric; they should not be scored as ordinary solvable contract tasks.

Run sandbox/schema/behavior checks. Provide a review sheet for an independent reviewer; keep review status explicit. Automated checks and a second pass by the same author are not independent annotation. Do not label the suite human-reviewed before the review occurs.

Deliverable: task specification, split manifest, draft data, automated validation, and review records. Resolve ambiguous cases before use.

## 3. Measure existing adapters before retraining

Run the existing 1.5B and 3B adapters on the same open development tasks using their original input format and generation settings. Compare prompted 7B only as a separately identified system: its prompting/training differs, so this is not a pure model-size experiment.

Primary parser metrics:

- Fraction of supported tasks whose correct implementations pass the generated contract.
- Fraction of targeted wrong outputs accepted, reported per task as well as by error family.
- Invalid output, semantic inconsistency, and escalation rates.
- Literal/column/operator/condition errors by family.
- Total parser calls, tokens, latency, and tool failures.

A contract that accepts the reference but also accepts the wrong programs is not successful. Structural exact match and teacher-forced loss are secondary metrics.

## 4. Test information grounding before a new training run

Use a small paired pilot on the open development set. Provide both existing adapters with a standardized, deterministic context block computed from the input CSV: column types, missing counts, supported category values/counts, and relevant numerical statistics with full precision. State null handling and tie behavior. Keep a strict context budget and explicitly flag unavailable information instead of silently omitting or truncating a required value.

This pilot isolates whether supplying information helps existing weights. Record the richer input as a new variant; compatibility with an existing adapter does not imply it was trained on that context distribution. Do not assume the original p1-v2 adapters already know the ft-t1 tool protocol.

For conditional or grouped statistics, a full-column median is not a substitute. If the input block cannot provide the required scope reliably, use a bounded read-only tool interface in the subsequent trained variant. The computation may use the user's request, schema and input data, never the gold contract, reference output or evaluation label.

Existing code to audit and reuse:

- [DataToolsFT](../src/igt/agents/tool_parser_ft.py) computes full-precision profiles and supports a row condition.
- [Tool-conversation builder](../finetune/data_gen/build_tools_data.py) already converts median/mode examples into tool conversations.
- [Contract pre-check](../src/igt/contracts/precheck.py) provides some schema/data consistency signals.

The current v3-tools set has only 73 training tool conversations and 6 validation tool conversations. It inherits the v2 coverage gaps; simply training it unchanged is not the recommended fix.

Treat pre-checks as fallible diagnostics. An absent category or zero matching rows can be legitimate, and choosing the wrong existing column can evade schema checks. Validate each rule on positive and negative development examples before making it a hard rejection.

Deliverable: a grounding pilot comparison and a decision on the smallest adequate production input format. Do not add tools solely because they already exist.

## 5. Build a revised training set around the selected input format

Retain useful checked v2 examples, but repair the information mismatch. Every target literal must be stated in the request or supported by a supplied statistic/tool observation. Recompute statistics from the actual input data; specify tie, null, group, and precision behavior.

Expand ordered categories, datetime conversion, mappings, scoped preservation, composite keys, multi-edit requests, and superficially similar columns. Aim initially for tens of distinct scenarios per difficult family across several training schemas, rather than thousands of paraphrases of one scenario. Set final counts after checking diversity and coverage.

Use varied data instances so the same schema/request can require different correct statistic values. Include unnecessary-tool cases, tool failures and recovery, and unsupported requests if the selected protocol supports explicit escalation. Keep those outputs consistent with the inference parser and outcome schema.

If tool use is selected, train multi-turn calls, observations and final contracts with the same inference protocol. Mask user/tool-observation tokens and train assistant actions plus final-contract tokens. Review masking, context length, and answer boundaries before a GPU job. Do not convert multi-turn data with a function that supervises only the final assistant message unless that is the explicitly chosen objective.

Rebuild validation across multiple held-out schemas with coverage of all difficult families. Evaluate the complete designated semantic set instead of a random subset that can omit rare patterns. Supervised training must exclude both development and locked-selection tasks/data.

Deliverable: new SFT data, coverage report, leakage/duplicate checks, semantic checks, and reviewed examples. Original v2 data remains available for reproduction.

## 6. Run one controlled fine-tuning experiment first

Start with 1.5B unless the open-development comparison gives a clear reason to prefer the existing 3B model. Hold the method fixed and change model size only in a separate comparison. There is no present evidence that raising the learning rate, LoRA rank, or epoch count is the first remedy.

Use explicit versioned data paths, pinned model/tokenizer revisions and package versions. Record resolved Trainer device count, world size, effective batch size, optimizer steps, warmup steps, seed, token-length distributions, dropped examples and loss masking checks. Save checkpoints to a new output directory and never overwrite v1 adapters.

Run a short smoke job to check data loading, loss masking, tool formatting if applicable, adapter saving/reloading, and actual free-generation outputs. Then run the full job. Select among a small, declared set of checkpoints using semantic open-development results, with cost and failure-family results visible. Loss is a monitoring signal, not the sole selection criterion.

If needed, train 3B with the same data/input protocol in a second experiment. Defer fine-tuned 7B until the smaller-model evidence justifies the extra cost.

## 7. Select the method, then test the complete pipeline

Predeclare the screening rule before generating new results. A recommended rule is:

- Require more correct implementations to pass than the original parser.
- Require no increase in observed wrong-output acceptance on the same reviewed task/candidate set.
- Require valid tool/contract behavior and report any loss of coverage or added cost.
- If no variant improves the trade-off, do not declare a winner; retain v1 or continue development with a new selection set if the old one has been inspected.

These are finite-sample screening requirements, not statistical non-inferiority guarantees. Report paired task-level differences and uncertainty; multiple wrong candidates from one task are clustered, not independent tasks.

After open-development selection, evaluate the frozen candidate on the 40 locked tasks once under the declared selection procedure. A failure to meet the rule is a useful result. Do not repeatedly edit the method against this set while calling it locked.

Run the selected parser through the generator, verifier and repair loop on development/selection tasks. Keep generator and repair models/budgets fixed so parser changes are interpretable. Record correct accepted outputs, wrong acceptances, abstentions, correct candidates withheld, useful repairs, and full cost including parser inference. Also check the no-contract-to-generator ablation: improved contract scores need not imply improved generation conditioning.

Deliverable: a selection record, versioned selected adapter/protocol, and an end-to-end development report. Proceed only if there is an operationally useful improvement; tool validity or lower loss alone is insufficient.

## 8. Evaluate independently after freezing the method

Create the larger, reviewed held-out benchmark after the training/selection procedure is fixed, using diverse datasets and distinct task instances. Run all comparison systems on the same tasks and report v1 and v2 separately. The original 100 test tasks may be rerun for diagnostics, but improvements on them are not fresh confirmatory evidence.

The core research claim must match the outcome: better accepted-correct completion, fewer incorrect acceptances, or a quantified trade-off. If the new method still does not beat direct generation on accuracy, report that result with its coverage and reliability implications.

## Immediate next deliverables

1. A concrete semantic-development specification and split/coverage rules.
2. Reviewed or explicitly draft tasks with positive and negative program examples.
3. A CPU validation/scoring harness.
4. A small Kaggle parser-comparison/grounding pilot using the saved adapters.

Only after these should we produce the new full-training notebook. This ordering addresses the information and validation problems before spending another long GPU session on fine-tuning.
