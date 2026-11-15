# Files to use for the project report

Use this guide for writing. Raw run bundles and executed notebooks remain in their canonical folders as supporting evidence; their presence does not mean every experiment should be a main result.

## Project definition and methods

- [Course proposal](course/SSSS_LMA_PROJECT_PROPOSAL.pdf) and [project handout](course/handout.txt).
- [Existing report text](reports/mid_submission_report.md) and [submitted PDF](reports/mid_submission_report.pdf). These describe an earlier project state; update experimental claims for the final report.
- [Contract schema](contract_schema.md), [evaluation guide](eval_set_and_contract_guide.md), and [verdict format](verdict_format.md).
- Implementation: `src/igt/` contains the verifier, contracts, agent, tools, sandbox and model interface. `eval/` contains experiment-specific runners. Keep source paths stable because source hashes and reproducibility checks depend on them.

Describe the intent parser as a role: selective facts computes relevant information before the parser; the trained tool parser chooses dataset queries before emitting a contract. Both current dual workflows use method-specific selected 1.5B adapters. Repair tools are separate from parser tools.

## Core experiments already completed

### Stage 1, training and agent groundwork

- [Stage 1 results](../results/STAGE1_RESULTS.md).
- [Fine-tuning results](../results/FINETUNE_RESULTS.md) and [training audit](../results/FINETUNE_AUDIT.md).
- [Agent results](../results/STAGE2_AGENT_RESULTS.md).
- [Prompt development](../results/STAGE2_DEV_RESULTS.md).

These establish the original pipeline and evaluation history. Keep their own configurations and task splits attached when citing them.

### Original six-system Stage 2 evaluation

- [Summary CSV](../results/stage2_clean_v1/stage2_summary.csv).
- [Complete metrics and paired comparisons](../results/stage2_clean_v1/stage2_metrics_test.json).
- [Failure analysis](../results/STAGE2_FAILURE_ANALYSIS.md).
- Original run bundles: `kaggle_outputs/stage2/clean_v1/`.
- Executed notebooks: `notebooks/executed/stage2/clean_v1/`.

The historical direct baseline gives 78 correct, 20 incorrect accepted, two withheld; historical proposed gives 76 correct, two incorrect accepted, 22 withheld. This is the matching historical six-system comparison. Do not substitute results from a later protocol into that table without identifying the configuration change.

### Current 3B FP16 generation and repair

- [Complete comparison](../results/stage2_dual_3b_analysis/ANALYSIS.md).
- [Strict collector summary](../results/stage2_dual_3b_analysis/comparison/summary.json).
- [All unsuccessful cases](../results/stage2_dual_3b_analysis/failure_review/INVENTORY.md).
- Original bundles: `kaggle_outputs/stage2/dual_parser_3b_fp16_v1/`.
- Executed notebooks: `notebooks/executed/stage2/dual_parser_3b_fp16_v1/`.

Selective facts and tools each give 73 correct, two incorrect accepted and 25 withheld on 100 tasks. Their correct task sets differ. Both generator and repair are unquantized Qwen2.5-3B-Instruct; the parsers use the selected 1.5B NF4 adapters. Tools uses 17.7% more total tokens for equal overall correctness. The remaining matched baseline runs for this newer workflow are not established by the old direct-baseline result.

### Repair-model comparison: 7B NF4 (archived)

Not used in the final report; moved to `archive/stage2_dual_7b/` on 2026-11-18.

- [Complete 7B analysis](../archive/stage2_dual_7b/results/stage2_dual_7b_analysis/ANALYSIS.md).
- [Paired 3B/7B evidence](../archive/stage2_dual_7b/results/stage2_dual_7b_analysis/repair_model_comparison.json).
- Original bundles: `archive/stage2_dual_7b/kaggle_outputs/dual_parser_v1/`.
- Executed notebooks: `archive/stage2_dual_7b/notebooks/executed/dual_parser_v1/`.

Both methods give 78 correct, one incorrect accepted, 21 withheld. Parser outputs and initial code match the corresponding 3B repair runs. Six tasks improve, one regresses, giving five additional correct results. The comparison changes repair size and precision (3B FP16 versus 7B NF4), so describe it accordingly. The initial generator is 3B FP16 in both.

## Parser-development experiments

- [Selective-facts training](../results/selective_facts_analysis/v1/ANALYSIS.md).
- [Scope fix](../results/selective_facts_analysis/scopefix1/ANALYSIS.md).
- [Completed normalization/correction comparison](../results/selective_facts_analysis/correction1/completed/ANALYSIS.md): 85 control, 88 normalized, 88 one-correction; zero accepted tested wrong outputs in each arm.
- [Tool-adapter training](../results/tool_use_v2_analysis/train_1_5b/ANALYSIS.md): selected epoch-2 parser result is 90 correct references, one accepted tested wrong output, 89 joint successes under its development criteria.
- [Base-model tool comparisons](../results/tool_use_v2_analysis/base_comparison/ANALYSIS.md).

These are parser-development checks with different tasks/metrics from end-to-end Stage 2. A parser reference-pass score is not final transformation accuracy. Use the analyses' explicit denominators and wrong-output probe definitions rather than mixing their percentages with Stage 2 results.

The earlier profile-input and untrained-tool diagnostics are useful in an ablation/history discussion, not extra main methods. Their reports are indexed in `docs/uploaded_artifacts.md`.

## Current development and future work

- [3B improvement plan](stage2_3b_improvement_plan.md).
- [Repair-development instructions](stage2_repair_dev.md).
- [Repair-development CPU validation](stage2_repair_dev_validation.json).
- Current notebook and source ZIP: `notebooks/kaggle/stage2_repair_dev/`.

The [completed 17-case, four-arm GPU repair experiment](../results/stage2_repair_dev_analysis/v1/ANALYSIS.md) gives control 11 correct, one incorrect accepted, five withheld; combined 16 correct, zero incorrect accepted, one withheld, with 26.1% fewer recorded model tokens. All 68 saved response sequences pass CPU replay. These are measured repair-only synthetic development results, not end-to-end Stage 2 accuracy. The [combined repair integration](stage2_combined_3b.md) is implemented for both parser methods; its [completed end-to-end analysis](../results/stage2_combined_3b_analysis/ANALYSIS.md) gives 74 correct, one incorrect accepted and 25 withheld for both methods, with 3.5%/8.4% more tokens than original 3B. This is a modest diagnostic improvement, with one repair regression. Parser correction and information-routing changes remain planned.

Other limitations for the report: repeated inspection of the 100 test cases; semantic errors despite schema-valid contracts; possible contradictions between edit and preservation requirements; precision changes in the repair comparison; and incomplete newer matched-baseline evaluation. See [open issues](open_issues.md) and the per-experiment analyses.

## What the cleanup removed

- Four adapter-containing Kaggle upload ZIPs whose members were verified against preserved canonical source/training bundles. Rebuild commands remain in `kaggle_uploads/`.
- Duplicate extracted notebook distributions, retaining the canonical ZIPs.
- An extracted selective-facts source tree. Four historical files differing from the canonical ZIP were preserved in `archive/notebooks/training/selective_facts_extracted_source_delta.zip`; the cleanup manifest records how to reconstruct each removed file.
- Disposable Python bytecode and pytest caches outside `.venv/`.

No original uploaded run, executed notebook, trained adapter, scored result or unique benchmark was deleted. The historical `pool_v3` is intentionally retained: five files differ from the current pool, so it is not a duplicate. Source and environment remain available for verification. The previous long README is preserved under `docs/history/README_before_report_cleanup_2026-11-15.md` for historical context.

[Machine-readable cleanup manifest](../archive/report_cleanup_manifest_2026-11-15.json) records removed file hashes, reasons and recovery locations. [Uploaded artifact index](uploaded_artifacts.md) records original upload names and canonical paths.

## Combined 3B gold-contract diagnostic

[Audited gold result](../results/stage2_combined_gold_analysis/ANALYSIS.md): 84 correct, zero incorrect accepted and 16 withheld, compared with matched predicted-contract runs at 74/1/25 each. Gold removes losses of initially correct outputs, while 16 initially incorrect programs remain unrepaired. Gold changes both generation and checking, and is not a deployable system. The historical 94/100 gold run uses a different repair configuration and initial programs. The final notebook collector error is a JSON mapping-key representation bug; all 100 inference records completed and passed offline validation with the documented compatibility fix.

## Contract-check and one parser correction

[Completed analysis](../results/contract_check_v1/completed/ANALYSIS.md): selective facts improves matched combined control 74/1/25 → 76/1/23 with no lost final successes; tools remains 74/1/25. Total tokens fall 12.4% and 8.2%. First contracts reproduce controls 100/100 each. Most correction replies are unchanged, the shared wrong acceptance persists, and an overbroad dtype diagnostic has a demonstrated synthetic false positive. Treat this as failure-informed development evidence, not independent confirmation.

## Current-stack conventional Baseline 2

[Verified analysis](../results/baselines_hints_v1/BASELINE2_ANALYSIS.md): 75 correct, 11 incorrect accepted, 14 withheld; current proposed selective facts 82/2/16 and tools 80/1/19. Baseline 2 has no parser, contract, schema guard or contract-derived hints. Its fixed generic mean-shift check rejects 16/100 correct references and five recorded correct final candidates. CPU replay finds zero mismatches. Do not mix historical direct-generation scores into a matched current table: only 67/100 initial programs reproduce the historical direct trace.

## Current tool parser without generator contract conditioning

[Generator-contract ablation](../results/baselines_hints_v1/NO_CONTRACT_TOOLS_ANALYSIS.md): no-contract/tools 78 correct, one incorrect accepted, 21 withheld, versus matched proposed tools 80/1/19. Parser/correction records match across all 100 tasks; only generator contract conditioning changes by design. No-contract saves 13.5% total tokens, but loses two net correct outcomes. The parser contract still controls checking and repair; the shared wrong-column acceptance persists, including a trace where correct initial code is changed to satisfy the wrong contract.

## Current selective-facts generator-contract ablation

[Selective ablation analysis](../results/baselines_hints_v1/NO_CONTRACT_SELECTIVE_ANALYSIS.md): 79 correct, two incorrect accepted, 19 withheld, versus matched proposed selective 82/2/16. Parser/correction records match on all 100 tasks. Omitting the contract from generation saves 12.1% total tokens but loses three net correct completions. Both variants retain contracts for verification and repair. These are diagnostic results on previously inspected tasks.
