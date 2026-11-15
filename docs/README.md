# Documentation: start here

This folder holds the project's written documentation. Results live in [`../results/`](../results/), not
here. If you only have a few minutes, read the files in **section 1** in order.

---

## 1. Start here (read in this order)

| # | File | What it gives you |
|---|---|---|
| 1 | [../README.md](../README.md) | Research question, headline Stage 2 table, where each report section's evidence is |
| 2 | [system_design.md](system_design.md) | Architecture, every module, design decisions and why, plan vs what was done |
| 3 | [FILE_INDEX.md](FILE_INDEX.md) | "If you need X, open Y": paths to every report, result, script and notebook |
| 4 | [REPORT_GUIDE.md](REPORT_GUIDE.md) | Which run backs which number in the report |
| 5 | [REPO_GUIDE.md](REPO_GUIDE.md) | What every folder is for, and what was moved to `archive/` and why |

The final reports are in the repo root: [`../ssss_report.pdf`](../ssss_report.pdf) and
[`../ssss_detailed_report.pdf`](../ssss_detailed_report.pdf).

---

## 2. Method specifications (how the system works)

| File | Content |
|---|---|
| [contract_schema.md](contract_schema.md) | The intent-contract vocabulary: 11 predicates, conditions, relation functions, dtype families |
| [verdict_format.md](verdict_format.md) | Verifier output: PASS / VIOLATION / UNCERTAIN and the failure records |
| [eval_set_and_contract_guide.md](eval_set_and_contract_guide.md) | How the 120 tasks and gold contracts were written (authoring rules) |
| [error_taxonomy.md](error_taxonomy.md) | The 22 error kinds used to label wrong LLM programs |
| [finetune_data_guide.md](finetune_data_guide.md) | How the fine-tuning data was generated and checked |

## 3. Course material and earlier reports

| File | Content |
|---|---|
| [course/SSSS_LMA_PROJECT_PROPOSAL.pdf](course/SSSS_LMA_PROJECT_PROPOSAL.pdf), [course/proposal.txt](course/proposal.txt) | Our approved proposal |
| [course/LMA_Group_Project_v1.pdf](course/LMA_Group_Project_v1.pdf), [course/handout.txt](course/handout.txt) | Course project handout |
| [reports/mid_submission_report.pdf](reports/mid_submission_report.pdf), [reports/mid_submission_report.md](reports/mid_submission_report.md) | Mid-submission report (earlier project state) |

## 4. Experiment run guides (Stage 2 and parser experiments)

One guide per experiment: what it changes, how to run it, and (for completed runs) its results. They
are listed in the order they were run. **Bold = its numbers appear in the final report.**

| Order | File | Experiment |
|---|---|---|
| 1 | **[stage2_six_notebooks.md](stage2_six_notebooks.md)** | Initial six-system Stage 2 comparison (78/20/2 direct vs 76/2/22 proposed) |
| 2 | [stage2_independent.md](stage2_independent.md), [stage2_independent_v2.md](stage2_independent_v2.md) | Stage 2 with contracts generated inside each run |
| 3 | **[selective_facts_training.md](selective_facts_training.md)** | Selective-facts parser: training and run instructions (Experiment 6b) |
| 4 | [selective_facts_scopefix.md](selective_facts_scopefix.md), [selective_facts_correction.md](selective_facts_correction.md) | Selective-facts fixes: router scope fix, normalisation and one correction |
| 5 | **[tool_use_v2.md](tool_use_v2.md)** | Tool-use parser (Experiment 6b) |
| 6 | [stage2_tools.md](stage2_tools.md) | Stage 2 with the parser-tool version |
| 7 | [stage2_dual_3b.md](stage2_dual_3b.md) | Stage 2 with unquantised 3B generation and repair |
| 8 | [stage2_repair_dev.md](stage2_repair_dev.md) | 3B repair development comparison |
| 9 | [stage2_combined_3b.md](stage2_combined_3b.md) | Improved (combined) 3B repair loop |
| 10 | [stage2_combined_gold.md](stage2_combined_gold.md) | Combined 3B repair with gold contracts |
| 11 | [stage2_contract_check.md](stage2_contract_check.md) | Contract self-check with one correction |
| 12 | **[stage2_repair_hints.md](stage2_repair_hints.md)** | **Final configuration**: repair hints (82/2/16, 80/1/19, gold 92/0/8) |
| 13 | **[stage2_baselines_hints.md](stage2_baselines_hints.md)** | Baselines re-run on the final stack (Baseline 2: 75/11/14; no-contract ablations) |
| — | [contract_quality/RUNBOOK.md](contract_quality/RUNBOOK.md) | Contract-quality pilot runbook |
| — | [stage2_3b_improvement_plan.md](stage2_3b_improvement_plan.md) | Plan that led to experiments 7–12 |

The `*_validation.json` files next to these guides are machine-written checks of each run's records.

## 5. Status, limits and future work

| File | Content |
|---|---|
| [open_issues.md](open_issues.md) | Known issues (e.g. OI-7 vocabulary gap, OI-8 unreviewed gold contracts) |
| [future_work.md](future_work.md) | Ideas not yet done |
| [uploaded_artifacts.md](uploaded_artifacts.md) | Index of files uploaded to / downloaded from Kaggle |

## 6. History (superseded plans; kept for the record)

[history/](history/) holds the original implementation roadmap (phases and gates), the extension
plans, the Phase 1 verifier brief, the contract-quality improvement plan, Stage 2 development notes and
an old README snapshot. They describe earlier states of the project; current facts are in sections 1–5.

Older experiments that the report does not use are in [`../archive/`](../archive/README.md).
