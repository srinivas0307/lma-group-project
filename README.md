# Intent-Gated Data Transformation Agent

Pandas code generation with an intent-contract parser, deterministic verification and bounded tool-assisted repair. Team SSSS · LMA Group Project 2026.

**Research question:** does an intent contract that states what must change and what must be preserved catch silent semantic errors in LLM-generated data transformations better than conventional data-quality checks?

**New here? Start with [docs/README.md](docs/README.md)**: a reading order and an index of every document. Then:
[docs/system_design.md](docs/system_design.md) (architecture and design decisions),
[docs/FILE_INDEX.md](docs/FILE_INDEX.md) (where every file is) and
[docs/REPO_GUIDE.md](docs/REPO_GUIDE.md) (what every folder is for, and the 2026-11-18 cleanup).

## Deliverables

| | File |
|---|---|
| Final report | [ssss_report.pdf](ssss_report.pdf) (source: [report_acl/final_report_short.tex](report_acl/final_report_short.tex)) |
| Detailed report | [ssss_detailed_report.pdf](ssss_detailed_report.pdf) (source: [report_acl/final_report.tex](report_acl/final_report.tex)) |
| Proposal and handout | [docs/course/](docs/course/) |
| Mid-submission report | [docs/reports/mid_submission_report.pdf](docs/reports/mid_submission_report.pdf) |

## Headline result (Stage 2, 100 test tasks: correct / wrong accepted / withheld)

| System | Result | Evidence |
|---|---|---|
| Baseline 1, direct generation | 78 / 20 / 2 | [stage2_clean_v1](results/stage2_clean_v1/stage2_summary.csv) |
| Proposed, initial | 76 / 2 / 22 | [stage2_clean_v1](results/stage2_clean_v1/stage2_metrics_test.json) |
| Baseline 2 + repair (final config) | 75 / 11 / 14 | [BASELINE2_ANALYSIS.md](results/baselines_hints_v1/BASELINE2_ANALYSIS.md) |
| **Proposed, final (selective facts)** | **82 / 2 / 16** | [selective_paired.json](results/repair_hints_v1/selective_paired.json) |
| **Proposed, final (tool use)** | **80 / 1 / 19** | [tools_paired.json](results/repair_hints_v1/tools_paired.json) |
| Gold contract, final | 92 / 0 / 8 | [gold_paired.json](results/repair_hints_v1/gold_paired.json) |

The final configuration was developed after inspecting the same test tasks, so its gains are diagnostic (see the report's Limitations).

## Where each report section's evidence lives

| Report section | Results | Run notebooks / raw outputs |
|---|---|---|
| Dataset: tasks, pool, scenarios, red team | [benchmark/tasks/INDEX.md](benchmark/tasks/INDEX.md), [data/README.md](data/README.md), [docs/eval_set_and_contract_guide.md](docs/eval_set_and_contract_guide.md) | `benchmark/`, `data/` |
| Contracts and verifier | [docs/contract_schema.md](docs/contract_schema.md), [docs/verdict_format.md](docs/verdict_format.md) | `src/igt/contracts/`, `src/igt/verifier/` |
| Exp 1–4, 6 and ablations A1–A6: Stage 1 verification | [results/STAGE1_RESULTS.md](results/STAGE1_RESULTS.md), `results/stage1/` | [notebooks/executed/stage1/](notebooks/executed/stage1/), `kaggle_outputs/generation/`, `kaggle_outputs/parsing/` |
| Exp 6: fine-tuning (QLoRA) | [results/FINETUNE_RESULTS.md](results/FINETUNE_RESULTS.md), [results/FINETUNE_AUDIT.md](results/FINETUNE_AUDIT.md) | [finetune_qlora_full.ipynb](notebooks/executed/training/finetune_qlora_full.ipynb), [finetune/README.md](finetune/README.md) |
| Exp 6b: selective-facts and tool-use parsers | `results/selective_facts_analysis/`, `results/tool_use_v2_analysis/` | `notebooks/executed/training/selective_facts_v1/`, `notebooks/executed/tool_use_v2/`, `notebooks/executed/parsing/` |
| Contract quality | `results/contract_quality_v1/`, [docs/contract_quality/RUNBOOK.md](docs/contract_quality/RUNBOOK.md) | `notebooks/executed/contract_quality/` |
| Exp 5 and A7: agent evaluation | [results/STAGE2_AGENT_RESULTS.md](results/STAGE2_AGENT_RESULTS.md), `results/stage2_agent/` | [agent-eval.ipynb](notebooks/executed/agent/agent-eval.ipynb) |
| Exp 7, A8–A10: Stage 2 six-system comparison | [stage2_summary.csv](results/stage2_clean_v1/stage2_summary.csv), [STAGE2_FAILURE_ANALYSIS.md](results/STAGE2_FAILURE_ANALYSIS.md) | `notebooks/executed/stage2/clean_v1/`, `kaggle_outputs/stage2/clean_v1/` |
| Stage 2 final configuration | `results/repair_hints_v1/`, [docs/stage2_repair_hints.md](docs/stage2_repair_hints.md) | `notebooks/executed/stage2/repair_hints_v1/`, `kaggle_outputs/stage2/repair_hints_v1/` |
| Final-config Baseline 2 and no-contract ablations | `results/baselines_hints_v1/` ([B2](results/baselines_hints_v1/BASELINE2_ANALYSIS.md), [selective](results/baselines_hints_v1/NO_CONTRACT_SELECTIVE_ANALYSIS.md), [tools](results/baselines_hints_v1/NO_CONTRACT_TOOLS_ANALYSIS.md)) | `notebooks/executed/stage2/baselines_hints_v1/` |
| Exp 8: safety | [results/safety/redteam_table.md](results/safety/redteam_table.md) | `benchmark/redteam/`, `src/igt/sandbox/` |

The intermediate Stage 2 runs that lead to the final configuration (`dual_parser_3b_fp16_v1`, `combined_3b_v1`, `combined_gold_v1`, `contract_check_v1`, `repair_dev_v1`, and earlier `independent`/`selective`/`tools` runs) stay in `results/`, `kaggle_outputs/stage2/` and `notebooks/executed/stage2/`. The final runs are paired against them, and the analysis scripts in `tools/` read them. [docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md) describes each one.

## Code

- `src/igt/`: verifier, contracts, parser and repair agents, tools, sandbox, model interface.
- `eval/`: experiment runners (one package per Stage 2 experiment; later ones import earlier ones).
- `finetune/`: data generation and QLoRA training for the parsers.
- `tools/`: analysis and audit scripts that produce the files in `results/`.
- `tests/`: unit and regression tests.
- `notebooks/kaggle/`, `notebooks/colab/`: runnable notebooks with sealed source packs; `kaggle_uploads/`: builders for the adapter datasets the Kaggle runs attach.
- `app/`: small demo UI (`requirements-demo.txt`).
- `archive/`: superseded experiments and history (see [archive/README.md](archive/README.md)).

## Local verification

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q tests/verifier tests/sandbox tests/test_data_checksums.py
./reproduce_all.sh   # recomputes the Stage 1 and safety results on CPU; no GPU inference
```

GPU runs were done on Kaggle (T4 x2). Their outputs are stored in `kaggle_outputs/` and the executed notebooks in `notebooks/executed/`. [Uploaded artifact index](docs/uploaded_artifacts.md).
