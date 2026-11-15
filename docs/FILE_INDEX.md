# File index for the evaluation

Quick lookup: **"if they ask about X, open Y"**. All paths are relative to the repo root and were checked
to exist on 2026-11-19. In VS Code, press **Ctrl+P** and paste a path to open it instantly.

---

## 0. Open these first (keep them in tabs during the viva)

| What | Path |
|---|---|
| Final report (PDF) | `ssss_report.pdf` |
| Detailed report (PDF) | `ssss_detailed_report.pdf` |
| Stage 1 main results | `results/STAGE1_RESULTS.md` |
| Stage 2 failure analysis | `results/STAGE2_FAILURE_ANALYSIS.md` |
| Docs index (start here) | `docs/README.md` |
| System design | `docs/system_design.md` |
| Repo guide (what every folder is) | `docs/REPO_GUIDE.md` |

---

## 1. Reports, proposal and course documents

| What | Path |
|---|---|
| Short report source (LaTeX) | `report_acl/final_report_short.tex` |
| Detailed report source (LaTeX) | `report_acl/final_report.tex` |
| Proposal and course handout | `docs/course/` |
| Mid-submission report | `docs/reports/mid_submission_report.pdf` |
| Which result backs which report section | `docs/REPORT_GUIDE.md` |
| Original plan (phases, gates, architecture) | `docs/history/implementation_roadmap.md` |
| Known issues (OI-7 vocabulary gap, OI-8 gold contracts) | `docs/open_issues.md` |
| Kaggle uploads index | `docs/uploaded_artifacts.md` |

---

## 2. Research question, hypotheses, headline numbers

| What | Path |
|---|---|
| RQ in one line + headline Stage 2 table | `README.md` |
| RQ, H1, H2 in full | `report_acl/final_report.tex` (section "Research question and hypotheses") |
| Stage 1 headline table (recall / FRR / silent error rate) | `results/STAGE1_RESULTS.md` |
| All Stage 1 metrics as JSON | `results/stage1/summary/metrics_stage1.json` |
| Stage 2 initial six-system table | `results/stage2_clean_v1/stage2_summary.csv` |
| Stage 2 metrics + paired tests | `results/stage2_clean_v1/stage2_metrics_test.json` |
| Stage 2 final config (selective / tools / gold) | `results/repair_hints_v1/selective_paired.json`, `tools_paired.json`, `gold_paired.json` |

---

## 3. Eval set: datasets, tasks, ground truth

| What | Path |
|---|---|
| Datasets description | `data/README.md` |
| The 4 evaluation CSVs | `data/` |
| List of all 120 tasks (dev / test) | `benchmark/tasks/INDEX.md` |
| One task file (request, reference code, gold contract) | `benchmark/tasks/<dataset>/<task_id>.yaml`, e.g. `benchmark/tasks/titanic/titanic_miss_004.yaml` |
| How tasks and contracts were written (authoring rules) | `docs/eval_set_and_contract_guide.md` |
| Frozen candidate pool | `benchmark/pool/pool_v4.jsonl` |
| Pool integrity manifest | `benchmark/pool/manifest.json` |
| Frozen schema text shown in prompts | `benchmark/prompts/` |
| How correct / wrong is decided (output equality) | `eval/equivalence.py` |
| How the pool was built | `eval/build_pool.py` |
| Generation runs that made the pool | `kaggle_outputs/generation/` |
| Agent scenarios R / C / T | `benchmark/agent_scenarios/rct_v4.jsonl` |
| Should-escalate scenarios (E) | `benchmark/agent_scenarios/E/` |
| Safety scenarios (S) | `benchmark/agent_scenarios/S/` |
| Red-team unsafe programs | `benchmark/redteam/samples.yaml` |

---

## 4. Contracts and the verifier (core method)

| What | Path |
|---|---|
| Contract vocabulary specification (11 predicates) | `docs/contract_schema.md` |
| Verdict format (PASS / VIOLATION / UNCERTAIN) | `docs/verdict_format.md` |
| Contract pydantic schema | `src/igt/contracts/schema.py` |
| Contract normalisation | `src/igt/contracts/normalize.py` |
| Contract self-check | `src/igt/contracts/precheck.py` |
| Verifier entry point | `src/igt/verifier/intent_verifier.py` |
| Predicate checks (one function each) | `src/igt/verifier/predicates.py` |
| Row alignment (index, content fallback) | `src/igt/verifier/alignment.py` |
| Conditions / relation functions / value equality | `src/igt/verifier/conditions.py`, `functions.py`, `values.py` |
| Shared types (Verdict, failures) | `src/igt/types.py` |
| Run gold contracts over the pool | `eval/run_intent_oracle.py` |
| Oracle results | `results/stage1/oracle/` |

---

## 5. Baselines

| What | Path |
|---|---|
| **Baseline 1** results by dataset | `results/stage1/baseline1/by_dataset.json` |
| Baseline 1 executed notebooks | `notebooks/executed/stage1/` |
| **Baseline 2** code (conventional checks) | `src/igt/baselines/conventional_checks.py` |
| Baseline 2 config (grid, default thresholds) | `configs/baseline2_v4.yaml` |
| Baseline 2 tuned thresholds + dev scores | `results/stage1/baseline2/thresholds.json` |
| Baseline 2 verdicts (lenient / strict / default) | `results/stage1/baseline2/verdicts_baseline2*.jsonl` |
| Baseline 2 tuning script | `eval/baseline2.py` |
| Baseline 2 on the final stack (75 / 11 / 14) | `results/baselines_hints_v1/BASELINE2_ANALYSIS.md` |
| Baseline 2 repair without contract | `eval/stage2_baselines_hints/nocontract.py` |
| **Baseline 3** (prompted 7B) results | `results/stage1/baseline3/` |
| Baseline 3 prompt selection on dev | `results/stage1/summary/baseline3_selection.json` |
| Prompted parser code | `src/igt/agents/intent_parser.py` |
| Raw parser outputs (all prompts / sizes) | `kaggle_outputs/parsing/` |
| LLM-as-verifier baseline (implemented, not run) | `src/igt/baselines/llm_verifier.py` |
| One YAML per system configuration | `configs/systems/` (e.g. `stage1_baseline2.yaml`, `stage2_proposed.yaml`) |

---

## 6. Fine-tuning (capability: core)

| What | Path |
|---|---|
| Fine-tuning results (metrics, provenance) | `results/FINETUNE_RESULTS.md` |
| Fine-tuning audit (leakage, data checks) | `results/FINETUNE_AUDIT.md` |
| Fine-tuning overview | `finetune/README.md` |
| Training data guide | `docs/finetune_data_guide.md` |
| Training script (QLoRA) | `finetune/train_qlora.py` |
| Training script (tool-use parser) | `finetune/train_qlora_tools.py` |
| Training configs (rank, lr, epochs) | `finetune/configs/`, `configs/systems/finetune_1_5b.yaml`, `finetune_3b.yaml` |
| Training datasets (never used in eval) | `finetune/datasets/README.md` |
| Example generator | `finetune/data_gen/generate_examples.py` |
| Example checker (valid, no-op fails, 5-gram overlap) | `finetune/data_gen/check_examples.py` |
| Executed training notebook | `notebooks/executed/training/finetune_qlora_full.ipynb` |
| Fine-tuned Stage 1 results | `results/stage1/finetuned/` |
| Size ablation (prompted 3B, 1.5B) | `results/stage1/size_ablation/` |
| Contract quality per parser | `results/stage1/summary/contract_summary.json` |
| Contract metrics code (F1, exact match) | `eval/contract_metrics.py` |
| Selective-facts parser analysis | `results/selective_facts_analysis/`, `docs/selective_facts_training.md` |
| Tool-use parser analysis | `results/tool_use_v2_analysis/`, `docs/tool_use_v2.md` |
| Audit script | `tools/audit_finetune.py` |

---

## 7. Repair agent, tools, orchestrator (capabilities: agentic, tool calling)

| What | Path |
|---|---|
| Agent evaluation results (full / feedback / naive) | `results/STAGE2_AGENT_RESULTS.md` |
| Agent metrics JSON | `results/stage2_agent/agent_metrics_test.json` |
| Annotated agent traces (wins and losses) | `results/agent_examples/README.md` (+ `R_test_003.md`, `R_test_045.md`, `T_test_005.md`, `S_011.md`) |
| Executed agent-eval notebook | `notebooks/executed/agent/agent-eval.ipynb` |
| Repair agent code | `src/igt/agents/repair_agent.py` |
| State machine (legal transitions) | `src/igt/orchestrator/state_machine.py` |
| Memory (attempt history, repeated-code hash) | `src/igt/orchestrator/memory.py` |
| Limits (attempts, tool calls, tokens) | `src/igt/orchestrator/limits.py` |
| Failure report given to the agent | `src/igt/pipeline/failure_report.py` |
| Tool registry (allow-list, argument validation) | `src/igt/tools/registry.py` |
| The data tools | `src/igt/tools/data_tools.py` |
| Fault injection (T scenarios) | `src/igt/tools/faults.py` |
| Agent configs | `configs/systems/agent_full.yaml`, `agent_feedback.yaml`, `agent_naive.yaml` |
| Agent runner / metrics | `eval/runner_agent.py`, `eval/agent_metrics.py` |
| Render a trace | `eval/show_trace.py` |
| Scripted LLM backend (deterministic tests) | `src/igt/llm/client.py` |
| Repair hints: design + results | `docs/stage2_repair_hints.md` |
| Repair hints: recipes / lints | `eval/stage2_repair_hints/recipes.py`, `eval/stage2_repair_hints/lint.py` |
| Repair hints offline replay | `results/repair_hints_v1/replay.json` |

---

## 8. Stage 2 end-to-end (RQ2)

| What | Path |
|---|---|
| Stage 2 runner (presets per system) | `eval/runner_stage2.py` |
| Stage 2 metrics code | `eval/stage2_metrics.py` |
| Initial six-system results | `results/stage2_clean_v1/` |
| Initial executed notebooks | `notebooks/executed/stage2/clean_v1/` |
| Failure analysis (24 failures) | `results/STAGE2_FAILURE_ANALYSIS.md` |
| Failure audit data + script | `results/stage2_failure_analysis/audit.json`, `tools/analyze_stage2_failures.py` |
| Final config results | `results/repair_hints_v1/` |
| Final config executed notebooks | `notebooks/executed/stage2/repair_hints_v1/` |
| Final config raw outputs | `kaggle_outputs/stage2/repair_hints_v1/` |
| Baselines on the final stack | `results/baselines_hints_v1/`, `docs/stage2_baselines_hints.md` |
| No-contract-to-generator ablations | `results/baselines_hints_v1/NO_CONTRACT_SELECTIVE_ANALYSIS.md`, `NO_CONTRACT_TOOLS_ANALYSIS.md` |
| Intermediate analyses | `results/stage2_combined_3b_analysis/ANALYSIS.md`, `results/stage2_combined_gold_analysis/ANALYSIS.md` |
| Contract self-check experiment | `docs/stage2_contract_check.md` |
| Dev-split Stage 2 results | `results/STAGE2_DEV_RESULTS.md` |

---

## 9. Failure analysis and qualitative examples

| What | Path |
|---|---|
| Error taxonomy (22 kinds) | `docs/error_taxonomy.md` |
| Error labels per wrong candidate | `benchmark/annotation/error_kinds_pool_v4.csv` |
| Errors caught by kind, per system | `results/stage1/summary/error_kinds.md` |
| Breakdown script | `eval/error_kind_breakdown.py` |
| Stage 2 failure analysis | `results/STAGE2_FAILURE_ANALYSIS.md` |
| Agent traces | `results/agent_examples/` |
| Unlabelled trajectory sheets (planned human labels) | `benchmark/annotation/trajectories/` |

---

## 10. Safety

| What | Path |
|---|---|
| Red-team result table | `results/safety/redteam_table.md` |
| Red-team samples | `benchmark/redteam/samples.yaml` |
| AST safety checker | `src/igt/sandbox/ast_guard.py` |
| Sandbox executor | `src/igt/sandbox/executor.py` |
| Safety table script | `eval/safety_table.py` |
| Safety scenarios (S) | `benchmark/agent_scenarios/S/` |
| Example unsafe-request trace | `results/agent_examples/S_011.md` |

---

## 11. Reproducibility

| What | Path / command |
|---|---|
| Regenerate Stage 1 + safety (CPU, ~20 min) | `./reproduce_all.sh` |
| Install | `requirements-dev.txt` (`python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`) |
| Quick tests | `.venv/bin/python -m pytest -q tests/verifier tests/sandbox tests/test_data_checksums.py` |
| Tests folder | `tests/` |
| Kaggle notebooks with sealed source packs | `notebooks/kaggle/` |
| All executed notebooks | `notebooks/executed/` |
| Raw outputs from Kaggle | `kaggle_outputs/` |

---

## 12. Live demo commands

| Demo | Command |
|---|---|
| Show a silent error the conventional checks missed | `.venv/bin/python eval/demo.py --candidate missed` |
| Demo UI | `app/demo_app.py` (needs `requirements-demo.txt`) |
| Render an agent trace | `.venv/bin/python eval/show_trace.py results/stage2_agent/agent_full R_test_003` |

---

## 13. Example tasks worth knowing by name

| Task | Why it's useful in the viva | Path |
|---|---|---|
| `titanic_miss_004` | Wrong contract (median 37 vs 28) rejects correct code | `benchmark/tasks/titanic/titanic_miss_004.yaml` |
| `airbnb_filt_002` | Wrong column in contract accepts a 0-row output | `benchmark/tasks/airbnb/airbnb_filt_002.yaml` |
| `titanic_col_005` | "Raise ages below 1 to 1" parsed as "add 1 to every age" | `benchmark/tasks/titanic/titanic_col_005.yaml` |
| `adult_conv_002` | Contract uses `<50K` instead of `<=50K` | `benchmark/tasks/adult/adult_conv_002.yaml` |
| `airbnb_dedup_004` | Repair hint followed a wrong contract → new wrong accept | `benchmark/tasks/airbnb/airbnb_dedup_004.yaml` |
