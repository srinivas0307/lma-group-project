# Repository guide: cleanup report and what every file is for

Written 2026-11-18, after the evaluation cleanup. Part 1 records what was moved and why. Part 2 explains what each folder and file is for.

★ marks material to open during the evaluation: the report cites it directly, or it holds the headline numbers.

---

# Part 1: Cleanup report

## How the decision was made

1. **The final report is the reference.** I read [ssss_report.pdf](../ssss_report.pdf) and the detailed report ([report_acl/final_report.tex](../report_acl/final_report.tex), [final_report_short.tex](../report_acl/final_report_short.tex)), the [proposal](course/proposal.txt) and the [course handout](course/handout.txt).
2. **Every path the report cites was checked.** All exist except two, listed under "Problems found" below.
3. **Every number the report quotes was traced to a run.** The report uses these runs:
   - Stage 1 (`results/stage1/`).
   - Fine-tuning.
   - Agent evaluation.
   - Safety.
   - Contract quality.
   - Stage 2 six-system comparison (`stage2_clean_v1`: 78/20/2, 76/2/22 and the others).
   - Final configuration (`repair_hints_v1`: 82/2/16, 80/1/19, 92/0/8).
   - Final-configuration Baseline 2 and no-contract ablations (`baselines_hints_v1`: 75/11/14, 79/2/19, 78/1/21).
   - Exp 6b parser contracts (`combined_3b_v1`).
4. **Code dependencies were traced.** Each Stage 2 experiment package in `eval/` imports earlier ones:
   - The final `stage2_repair_hints` imports `stage2_contract_check`, `stage2_combined_gold`, `stage2_combined_3b`, `stage2_dual_3b`, `stage2_repair_dev`, `stage2_selective`, `stage2_tools` and `finetune/*`.
   - The final runs are paired against `combined_gold` (84/0/16) and `contract_check` (76/1/23, 74/1/25).
   - So these intermediate experiments **stay**, even though their own numbers are not in the report. Moving them would break the final pipeline, the analysis scripts and the tests.
5. **Only clean leaves were archived.** These are things nothing kept depends on and the report does not use.
6. **Tests were checked before and after.** I ran the whole suite with the pinned `requirements-dev.txt`:

   | | Passed | Failed | Collection errors |
   |---|---|---|---|
   | Before the cleanup | 672 | 16 | 10 |
   | After the cleanup | 672 | 16 | 8 |

   - The same 16 tests fail in both runs. They fail because GPU or training libraries are missing on the laptop, not because of the cleanup.
   - The 2 fewer collection errors are the two tests that moved into the archive with the code they test.

All moves used `git mv`, so file history is kept and contents are unchanged.

## Moved into `archive/` (not needed for the evaluation)

| What | From | To | Why |
|---|---|---|---|
| Fresh Stage 2 reimplementation: 20 cell files, 9 tests, 5 notebooks, 3 ZIPs, prompts, README | `stage2_fresh/` | `archive/stage2_fresh/` | A from-scratch rewrite of the pipeline. Its results (selective 71/2/27, tool use 68/8/24) are **not in the report** and are worse than the reported system. Nothing else imports it. |
| 7B-NF4 repair experiment ("dual parser v1", 78/1/21) | `eval/stage2_dual/`, `notebooks/kaggle/stage2_dual/`, `notebooks/executed/stage2/dual_parser_v1/`, `kaggle_outputs/stage2/dual_parser_v1/`, `results/stage2_dual_7b_analysis/`, `kaggle_uploads/stage2_dual/`, `tests/test_stage2_dual.py`, `tools/analyze_stage2_dual_7b.py`, `docs/stage2_dual.md`, `docs/stage2_dual_validation.json` | `archive/stage2_dual_7b/` (same sub-structure) | A side comparison of repair-model size that the report does not use. No kept code imports `eval.stage2_dual`; only its own test, notebook builder and upload builder did. |
| Gold contract + 7B repair | `eval/stage2_combined_gold_7b/`, `notebooks/kaggle/stage2_combined_gold_7b/`, `tests/test_stage2_combined_gold_7b.py`, `docs/stage2_combined_gold_7b.md`, `docs/stage2_combined_gold_7b_validation.json` | `archive/stage2_combined_gold_7b_not_run/` | **Never run** on GPU (the README said "GPU results are pending"). It has no results and is not in the report. |
| Superseded plan | `docs/stage2_post_combined_next_steps.md` | `archive/docs/` | A "what to do next" plan that the later contract-check and repair-hints work has already carried out. |
| One-off layout script | `tools/restructure.py` | `archive/tools/` | Used once on 2026-10-30 to reorganise folders. Not part of any experiment. |

## Brought back from `archive/` (needed for the evaluation)

| What | From | To | Why |
|---|---|---|---|
| ★ 7 Stage 1 executed notebooks: Baseline 1 on 3B and 1.5B, Baseline 3 parser runs v1/v2/v3, and candidate generation on 3B and 1.5B | `archive/notebooks/executed/stage1/` | `notebooks/executed/stage1/` | These are the GPU runs behind Experiments 1 and 4 and ablations A2, A4 and A6. `results/STAGE1_RESULTS.md` cites them. |
| ★ Agent evaluation notebook | `archive/notebooks/executed/agent/agent-eval.ipynb` | `notebooks/executed/agent/agent-eval.ipynb` | The report cites **this exact path** (Experiment 5 / A7), but the file existed only in the archive. |
| ★ Full fine-tuning notebook | `archive/notebooks/executed/training/finetune_qlora_full.ipynb` | `notebooks/executed/training/` | The QLoRA training run behind Experiment 6, the fine-tuning capability. |

## Other changes

- **Link fixes.** Markdown links and path mentions that pointed at moved files were updated in:
  - `README.md` and `docs/REPORT_GUIDE.md`.
  - `docs/uploaded_artifacts.md`, `docs/open_issues.md` and `docs/history/implementation_roadmap.md`.
  - `finetune/README.md`.
  - `results/STAGE1_RESULTS.md`, `results/FINETUNE_RESULTS.md` and `results/STAGE2_AGENT_RESULTS.md`.
  - The moved docs themselves.
- **`.gitignore`.** The ignored adapter ZIP path now points to `archive/stage2_fresh/kaggle_upload/tooluse_adapter_1_5b.zip`. The file itself moved with the folder.
- **`README.md`.** Rewritten as an evaluation entry point. It has the headline table and maps each report section to its evidence.
- **`archive/README.md`.** A new section records these moves.
- **`docs/REPORT_GUIDE.md`.** The 7B section is marked as archived.
- **Python bytecode caches.** 34 `__pycache__/*.pyc` files had been committed by mistake (`.gitignore` already excludes them). They were deleted while clearing caches before the test run. Python recreates them automatically.

## Problems found (not fixed; you should know before the eval)

1. **Missing script.** The report's Experiment 6b says the Stage 1 numbers for the new parsers (selective facts FRR 0.120, tool use 0.139) come from `tools/build_new_parser_stage1.py`.
   - That file has never existed in this repo or its git history.
   - Those numbers are not in any file under `results/`.
   - If asked, you cannot currently show where that table was computed.
2. **Report build script.** `report_acl/build.py` reads `final_report.md` from the repo root, which does not exist. The `.tex` files and PDFs are fine; only rebuilding from Markdown fails.
3. **Pre-existing broken links.** Some links in `docs/history/README_before_report_cleanup_2026-11-15.md` (an old copy of the root README) and one in `archive/distributions/` were already broken. They are historical and were left alone.

---

# Part 2: What every folder and file is for

## Root

| File | Purpose |
|---|---|
| ★ `ssss_report.pdf` | Final report (short version). |
| ★ `ssss_detailed_report.pdf` | Detailed final report, with all experiments, ablations and limitations. |
| ★ `README.md` | Entry point: headline results and a map from report sections to evidence. |
| `reproduce_all.sh` | Recomputes all Stage 1 and safety results on CPU from frozen inputs and checks they are identical. No GPU. |
| `requirements.txt` | Pinned core dependencies: pandas 2.3.3, numpy, pyyaml, pydantic. The pins must match Kaggle so labels agree. |
| `requirements-dev.txt` | Core dependencies plus pytest. |
| `requirements-demo.txt` | Gradio, for the demo in `app/`. |
| `requirements-vllm.txt` | Optional vLLM backend (`src/igt/llm/vllm_backend.py`). |
| `pyproject.toml` | Package and pytest configuration. |
| `.gitignore` | Excludes `.venv`, caches, trained adapter runs and large adapter ZIPs. |

## `report_acl/`: LaTeX source of the reports

| File | Purpose |
|---|---|
| `final_report.tex` | Source of the detailed report. |
| `final_report_short.tex` | Source of the short report. |
| `template.tex`, `figure.tex` | Page template and the TikZ pipeline figure. |
| `acl.sty`, `acl_natbib.bst` | ACL style files. |
| `build.py`, `filter.lua` | Markdown → LaTeX converter (pandoc). Needs `final_report.md`, which is missing (see Problems). |

## `docs/`: specifications and experiment write-ups

| File | Purpose |
|---|---|
| ★ `course/proposal.txt`, `course/SSSS_LMA_PROJECT_PROPOSAL.pdf` | Approved project proposal: research question, capabilities, plan. |
| `course/handout.txt`, `course/LMA_Group_Project_v1.pdf` | Course requirements (five capabilities, evaluation expectations). |
| `reports/mid_submission_report.pdf`, `mid_submission_report.md` | Mid-term submission (earlier project state). |
| ★ `contract_schema.md` | Full reference for the intent-contract format: every predicate and relation. |
| ★ `eval_set_and_contract_guide.md` | How the evaluation set and gold contracts were built, and the verifier semantics (§5). |
| `verdict_format.md` | Output format every verifier returns (PASS / VIOLATION / UNCERTAIN). |
| `error_taxonomy.md` | Error kinds used to label wrong candidates (failure analysis). |
| `finetune_data_guide.md` | How the parser fine-tuning data was generated, checked and reviewed. |
| ★ `REPORT_GUIDE.md` | Which result file backs which claim, and which experiments can or cannot be compared. |
| `REPO_GUIDE.md` | This file. |
| `stage2_six_notebooks.md` | How the six-system Stage 2 comparison (`clean_v1`, Exp 7) was run. |
| `stage2_dual_3b.md` | 3B FP16 generator/repair run with both new parsers (73/2/25): the first step towards the final configuration. |
| `stage2_repair_dev.md`, `stage2_3b_improvement_plan.md` | Repair-only development experiment and the improvement plan made from the 3B failure review. |
| `stage2_combined_3b.md` | Combined repair loop with both parsers (74/1/25). |
| `stage2_combined_gold.md` | Same loop with gold contracts (84/0/16): the baseline for the final gold run. |
| `stage2_contract_check.md` | Contract self-check plus one correction (76/1/23 and 74/1/25): part of the final configuration. |
| ★ `stage2_repair_hints.md` | **Final configuration**: repair hints, giving 82/2/16, 80/1/19 and 92/0/8. |
| ★ `stage2_baselines_hints.md` | Baseline 2 and the no-contract ablations re-run on the final configuration. |
| `selective_facts_training.md`, `selective_facts_scopefix.md`, `selective_facts_correction.md` | Selective-facts parser: training, router scope fix, and the normalization/correction step (Exp 6b). |
| `tool_use_v2.md` | Tool-use parser training (Exp 6b). |
| `stage2_independent.md`, `stage2_independent_v2.md`, `stage2_tools.md` | Earlier attempts at running Stage 2 with profile-based or tool-based parsers. These are listed as "not completed" in the report. |
| `contract_quality/RUNBOOK.md`, `contract_quality_improvement_plan.md` | Contract-quality experiment: how to run it, and the plan it came from. |
| `open_issues.md`, `future_work.md` | Known limitations and future-work ideas (feed the report's Limitations section). |
| `uploaded_artifacts.md` / `.json` | Index of every Kaggle upload: original name → location in the repo. |
| `report_cleanup_manifest_2026-11-15.json`, `*_validation.json` | Machine-readable records: an earlier cleanup, and CPU validation of notebook source packs. |
| `history/` | Old roadmap, plans and an earlier README. Historical only. |

## `src/igt/`: the system (★ the core implementation)

**Contracts and verification**

| File | Purpose |
|---|---|
| ★ `contracts/schema.py` | Intent-contract schema (vocabulary v1.0); validates gold and parsed contracts. |
| `contracts/normalize.py` | Canonical predicate keys, for comparing a predicted contract with the gold one (exact match, F1). |
| `contracts/precheck.py` | Checks a contract against the input data before use (extension). |
| ★ `verifier/intent_verifier.py` | **The intent verifier**: checks every contract predicate on before/after data and returns a verdict. |
| `verifier/predicates.py` | One check per predicate type (holds / violated / uncertain). |
| `verifier/alignment.py` | Matches output rows to input rows. |
| `verifier/conditions.py` | Evaluates row conditions on the input data (null semantics). |
| `verifier/functions.py` | Expected values for column relations (add, round, map, cast…). |
| `verifier/values.py` | Value equality and dtype families. |
| `types.py` | Shared `Verdict` and other types used by every verifier and system. |

**Agents (LLM components)**

| File | Purpose |
|---|---|
| ★ `agents/intent_parser.py` | Prompted intent parser (Baseline 3): request + schema → contract. |
| `agents/tool_parser.py`, `agents/tool_parser_ft.py` | Parsers that query the data with read-only tools first (prompted / fine-tuned). |
| `agents/reask.py` | Re-asks the parser once after an invalid contract (implemented, not run). |
| ★ `agents/code_generator.py` | Code-generator prompts: b1-v1 (request only), b2-v1 (with contract). |
| `agents/code_generator_v2.py` | b2-v2 prompt, the one used by the proposed system in Stage 2. |
| `agents/code_extract.py` | Pulls the `transform` function out of an LLM reply. |
| ★ `agents/repair_agent.py` | **Bounded tool-using repair agent** (a1-v4) and the naive/feedback agent baselines. |
| `agents/repair_agent_v5.py`, `agents/repair_agent_ext.py` | Opt-in prompt and loop variants (extensions). |

**Baselines**

| File | Purpose |
|---|---|
| ★ `baselines/conventional_checks.py` | **Baseline 2**: conventional data-quality checks (schema, nulls, uniqueness, mean shift…). |
| `baselines/llm_verifier.py` | LLM-as-verifier baseline (implemented, not run). |

**Safety, tools and orchestration**

| File | Purpose |
|---|---|
| ★ `sandbox/ast_guard.py` | AST allow-list safety checker: the first layer, blocks unsafe code before it runs. |
| ★ `sandbox/executor.py` | Runs generated code in a separate, limited process (CPU, memory and time limits; network stub). |
| `sandbox/environment.py`, `sandbox/docker_executor.py` | Sandbox environment probe; optional Docker backend. |
| ★ `tools/data_tools.py` | The 5 agent tools: inspect_dataset, profile_column, diff_datasets, run_transformation, run_verifier. |
| `tools/registry.py` | Tool registry: allow-listed tools, JSON argument schemas, structured errors (tool-call validity). |
| `tools/faults.py` | Fault injection for the agent-evaluation scenarios. |
| `orchestrator/state_machine.py` | Allowed pipeline states and transitions; an illegal step raises an error. |
| `orchestrator/limits.py` | All budgets in one place (3 attempts, 8 tool calls, token limit). |
| `orchestrator/memory.py` | Attempt history for the repair agent; detects repeated code. |
| `pipeline/failure_report.py` | One failure-report format for every way a candidate can fail. |
| `pipeline/trace_view.py` | Readable rendering of a repair run (case studies). |
| `llm/client.py`, `llm/hf.py`, `llm/vllm_backend.py` | One LLM interface with caching and token budget; transformers and vLLM backends. |
| `config.py`, `paths.py` | Per-system config loading; every project path defined in one place. |

## `eval/`: experiment runners and metrics

**Stage 1 (verification on the frozen candidate pool)**

| File | Purpose |
|---|---|
| `generate.py`, `label_candidates.py`, `build_pool.py` | Generate candidate code (GPU), label it against the reference, and freeze the candidate pool. |
| ★ `equivalence.py` | Answer key: is a candidate's output equal to the reference? |
| `baseline1_by_dataset.py` | Baseline 1 (no verification) metrics. |
| `baseline2.py` | Baseline 2: tune thresholds on dev, judge the pool. |
| `run_intent_oracle.py` | Intent verifier with gold contracts (oracle), plus the change-only / preserve-only ablation (A1). |
| `parse_contracts.py`, `parse_contracts_tools*.py` | Run a parser over all tasks (GPU). |
| `score_parsers.py`, `contract_metrics.py`, `metrics.py` | Score parser runs: contract metrics, verdicts, Stage 1 tables. |
| `ext_score.py` | Scores extension systems; the report cites it for Exp 6b. |
| `error_kind_breakdown.py`, `draft_error_kinds.py` | Recall by error kind (failure analysis). |
| `safety_table.py` | Red-team safety table (Exp 8). |

**Agent evaluation (Exp 5)**

| File | Purpose |
|---|---|
| `build_agent_scenarios.py` | Builds the R/C/T agent scenarios. |
| `runner_agent.py` | Runs an agent system over the scenarios. |
| ★ `agent_metrics.py` | Agent metrics. The report cites it. |
| `trajectory_labels.py`, `make_annotation_sheet.py`, `annotation_agreement.py` | Human-label sheets and agreement (prepared, not labelled). |

**Stage 2 end to end.** Each package is one experiment, and later ones import earlier ones.

| Folder / file | Experiment |
|---|---|
| `runner_stage2.py`, `stage2_metrics.py`, `runner_stage2_prompts.py` | Original Stage 2 runner, metrics and prompt development. |
| ★ `stage2_clean/` | Six-system comparison, Exp 7 (`clean_v1`). |
| `stage2_independent/`, `stage2_independent_v2/`, `stage2_selective/`, `stage2_tools/` | Earlier parser-in-the-loop attempts. Kept as code dependencies. |
| `stage2_dual_3b/` | 3B FP16 run with both new parsers. |
| `stage2_repair_dev/` | Repair-only development: evidence and schema guard. |
| `stage2_combined_3b/` | Combined repair loop. |
| `stage2_combined_gold/` | The same loop with gold contracts. |
| `stage2_contract_check/` | Contract self-check plus one correction. |
| ★ `stage2_repair_hints/` | **Final configuration**: recipes (`recipes.py`) and lints (`lint.py`) added to the repair prompt. |
| ★ `stage2_baselines_hints/` | Baseline 2 and the no-contract ablations on the final configuration. |
| `contract_quality/` | Contract-quality experiment. |

**Extensions and demo**

| File | Purpose |
|---|---|
| `ext_*.py`, `runner_*_ext.py`, `run_llm_verifier.py`, `with_sandbox.py`, `run_config.py` | Optional extensions: re-ask, pre-check, Docker, vLLM, LLM verifier, configs. Mostly "implemented, not run". |
| `demo.py`, `show_trace.py` | Command-line demo of the pipeline; prints a recorded repair run. |

## `tools/`: analysis and audit scripts (CPU, no model calls)

Each script reads a Kaggle output bundle and writes the analysis in `results/`:

| Script | Writes / checks |
|---|---|
| ★ `analyze_repair_hints_run.py`, `replay_repair_hints.py` | `results/repair_hints_v1/`: final results and CPU replay. |
| ★ `analyze_baselines_hints_run.py`, `audit_baseline2_hints.py`, `audit_no_contract_tools_hints.py` | `results/baselines_hints_v1/`. |
| `analyze_contract_check_run.py`, `audit_contract_check_runs.py`, `analyze_contract_diagnostics.py` | `results/contract_check_v1/`. |
| `analyze_stage2_combined_3b.py`, `validate_stage2_combined_3b.py` | `results/stage2_combined_3b_analysis/`. |
| `analyze_stage2_combined_gold.py`, `collect_stage2_combined_gold.py` | `results/stage2_combined_gold_analysis/`. |
| `analyze_stage2_dual_3b.py`, `analyze_stage2_3b_failures.py` | `results/stage2_dual_3b_analysis/`, including the failure inventory. |
| `analyze_stage2_repair_dev.py`, `validate_stage2_repair_dev.py` | `results/stage2_repair_dev_analysis/`. |
| `analyze_stage2_failures.py` | `results/STAGE2_FAILURE_ANALYSIS.md` (Exp 7 failures). |
| `analyze_stage2_independent_proposed.py`, `analyze_parser_diagnostic.py`, `analyze_stage2_tools.py` | Earlier Stage 2 attempts. |
| `analyze_selective_facts*.py`, `analyze_selective_correction_*.py` | `results/selective_facts_analysis/` (Exp 6b). |
| `analyze_tool_use_training.py`, `analyze_tool_use_base.py` | `results/tool_use_v2_analysis/` (Exp 6b). |
| `audit_finetune.py` | `results/FINETUNE_AUDIT.md`. |

## `finetune/`: parser fine-tuning (★ fine-tuning capability)

| File | Purpose |
|---|---|
| ★ `README.md` | How to train, and where results are. |
| ★ `train_qlora.py` | QLoRA training of the intent parser (Exp 6). |
| `train_qlora_tools.py` | QLoRA training of the tool-using parser variant. |
| `configs/*.yaml` | Training settings for 1.5B, 3B, tool variants and a CPU smoke run. |
| `data_gen/` | Builds training data: `prepare_datasets.py` → `generate_examples.py` → `check_examples.py` → `build_sft.py`, plus the hand-review scripts. |
| `datasets/` | 7 non-evaluation training CSVs (snapshots of `data/finetune/`), so training never sees test data. |
| `data/` | Generated training data: `sft_train.jsonl` and `sft_val.jsonl`, hand review in `review.csv`, plus `selective_facts_v1/` and `tool_use_v2/` sets. |
| `selective_facts/`, `selective_facts_scopefix/`, `selective_facts_correction/` | Selective-facts parser: training, router scope fix, and normalization/correction (Exp 6b). |
| `tool_use/` | Tool-use parser training and inference (Exp 6b). |
| `runs/` (not in git) | Trained LoRA adapters for 1.5B and 3B. Local only. |

## `benchmark/`: frozen evaluation inputs (★ the dataset)

| Path | Purpose |
|---|---|
| ★ `tasks/` + `INDEX.md` | 120 tasks: 4 datasets × 5 types × 6 tasks. Each has a request, a gold contract and a reference solution. |
| ★ `pool/` | Frozen candidate pool v4: generated programs with output pickles. Stage 1 runs on it. |
| `agent_scenarios/` | Agent-evaluation scenarios (`rct_v4.jsonl`, E and S sets). |
| `redteam/samples.yaml` | 27 unsafe programs for the safety evaluation. |
| `prompts/schema_b1-v1.json` | Frozen schema-prompt format. |
| `annotation/` | Error-kind labels and prepared second-annotation and trajectory-label sheets. |
| `contract_quality_v1/` | Contract-quality development suite. |

## `data/`: raw datasets

| File | Purpose |
|---|---|
| `adult_dev.csv`, `airbnb_dev.csv`, `Titanic-Dataset.csv`, `yellow_tripdata_2026-01.csv` | The 4 evaluation datasets. |
| `checksums.json` | Hashes checked by `tests/test_data_checksums.py`. |
| `README.md` | Dataset sources. |
| `finetune/` | 8 unrelated public datasets, used only to build fine-tuning data. |

## `configs/`

| File | Purpose |
|---|---|
| `baseline2_v4.yaml` | Baseline 2 thresholds (tuned on dev). The report cites it. |
| `systems/*.yaml` | One config per system and run (Stage 1, Stage 2, agents, parsers, fine-tuning, extensions), with the generator `make_configs.py`. |

## `results/`: every reported number (★)

| Path | Contents |
|---|---|
| ★ `STAGE1_RESULTS.md`, `stage1/` | Stage 1, Exp 1–4 and 6 and ablations A1–A6: per-system verdicts, metrics and summary tables. |
| ★ `FINETUNE_RESULTS.md`, `FINETUNE_AUDIT.md`, `finetune_audit/` | Fine-tuning results and audit of the training data and adapters. |
| ★ `STAGE2_AGENT_RESULTS.md`, `stage2_agent/`, `agent_examples/` | Agent evaluation (Exp 5, A7) and example trajectories. |
| ★ `stage2_clean_v1/`, `STAGE2_FAILURE_ANALYSIS.md`, `stage2_failure_analysis/` | Stage 2 six-system comparison (Exp 7, A8–A10) and its failure analysis. |
| ★ `repair_hints_v1/` | Final configuration: paired results for gold, selective and tools, plus CPU replay. |
| ★ `baselines_hints_v1/` | Final-configuration Baseline 2 (75/11/14) and no-contract ablations, with audits. |
| ★ `safety/` | Red-team table (Exp 8). |
| `contract_quality_v1/` | Contract-quality suite validation. |
| `selective_facts_analysis/`, `tool_use_v2_analysis/` | Parser development and screening (Exp 6b). |
| `contract_check_v1/`, `stage2_combined_3b_analysis/`, `stage2_combined_gold_analysis/`, `stage2_dual_3b_analysis/`, `stage2_repair_dev_analysis/` | Intermediate steps towards the final configuration. The final runs are paired against them. |
| `stage2_independent_analysis/`, `stage2_tools_analysis/` | Earlier Stage 2 attempts. |
| `STAGE2_DEV_RESULTS.md` | Prompt development on dev tasks: how the b2-v2 and a1-v4 prompts were chosen. |

## `notebooks/`

| Path | Purpose |
|---|---|
| ★ `executed/` | Notebooks exactly as they ran on Kaggle, with outputs. This is the GPU evidence. |
| `executed/stage1/` | Baselines 1/3 and generation. |
| `executed/agent/` | Agent evaluation. |
| `executed/training/` | Fine-tuning. |
| `executed/stage2/<run>/` | Each Stage 2 run. |
| `executed/parsing/`, `executed/tool_use_v2/`, `executed/contract_quality/` | Parser and contract-quality runs. |
| `kaggle/` | Runnable templates. |
| `kaggle/<experiment>/` | Each holds the notebooks, a `make_notebooks.py` builder, and a sealed `*_source.zip` with the exact code the run used. |
| `kaggle/stage2_0*.ipynb`, `stage2_six_notebooks.zip` | The six-system Exp 7 notebooks. |
| `colab/` | Colab versions of some runs. |
| `training/finetune_qlora_kaggle.ipynb` | Template for parser training. |

## `kaggle_outputs/`: raw run outputs downloaded from Kaggle (do not edit)

| Path | Contents |
|---|---|
| `generation/` | Candidate generations from 3B and 1.5B (pool source). |
| `parsing/` | Every parser run: 1.5B, 3B and 7B prompted p1-v1 to v3, fine-tuned 1.5B and 3B, and selective-facts variants. |
| `stage2/<run>/` | Output ZIP bundles of every Stage 2 run. `clean_v1`, `repair_hints_v1`, `baselines_hints_v1` and `combined_3b_v1` are the ones the report uses. |
| `training/`, `tool_use_v2/` | Training bundles for the new parsers. |
| `contract_quality/` | Contract-quality pilot outputs. |

## `kaggle_uploads/`

Builders for the adapter datasets that the final Kaggle runs attach. Needed to reproduce the final runs.

## `tests/`

| Path | Purpose |
|---|---|
| ★ `verifier/` | Predicate tests, and a test that every gold contract passes its own reference output. |
| ★ `sandbox/` | AST guard, executor and red-team tests (safety). |
| `agent/` | Repair agent, tools, LLM client and runner metrics. |
| `test_stage2_*.py` | One per Stage 2 experiment: pipeline smoke tests with scripted models, and source-pack checks. |
| `test_selective_*.py`, `test_tool_use_v2.py`, `test_finetune_*.py` | Parser training and data. |
| `test_data_checksums.py`, `test_types.py` | Datasets unchanged, shared types. |
| `extensions/` | Extension features. |

Run: `.venv/bin/python -m pytest -q tests`. Expect 672 passed. On a laptop without GPU libraries (torch/peft), 16 fail and 8 error during collection.

## `app/` and `sandbox/`

| Path | Purpose |
|---|---|
| `app/demo_app.py`, `app/demo_core.py` | Gradio demo of the pipeline (request → contract → code → verify → repair). Optional, but useful to show live. |
| `sandbox/docker/` | Dockerfile for the optional Docker sandbox. |

## `archive/`: history (not needed for the evaluation)

| Path | Contents |
|---|---|
| `stage2_fresh/`, `stage2_dual_7b/`, `stage2_combined_gold_7b_not_run/`, `docs/`, `tools/` | Moved today (Part 1). |
| `pools/`, `generation/`, `results/`, `configs/`, `agent_scenarios/` | Superseded pool and parser versions v1–v3. Some tests still read them, so **do not move them**. |
| `notebooks/` | Older templates and extension notebooks. Some tests read `notebooks/core/`. |
| `stage2/` | Early Stage 2 development, prompt development and test bundles. |
| `extensions/`, `review/`, `distributions/`, `import_duplicates/` | Extension results, fine-tuning review material, and older duplicates. |
| `README.md`, `cleanup_manifest_2026-11-13.json` | Archive index and earlier move records. |
