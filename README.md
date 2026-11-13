# Intent-Gated Data Transformation Agent

An agent that checks LLM-generated pandas transformations against an intent contract describing what must change and what must be preserved. A bounded repair loop uses tools to inspect failures, propose fixes, and escalate unresolved requests.

**Team SSSS · LMA Group Project 2026**

## Current status and next step

Stage 1 evaluation, parser fine-tuning, agent experiments, Stage 2 prompt development, and the final six-system Stage 2 evaluation are complete. All six Kaggle runs contain 100 verified task records, and the collector contains 600 records across the same 100 test tasks. Its 657 files match an independent local collection.

See the [Stage 2 summary](results/stage2_clean_v1/stage2_summary.csv), [full metrics and paired comparisons](results/stage2_clean_v1/stage2_metrics_test.json), and [import validation](kaggle_outputs/stage2/clean_v1/import_validation.json). Completed bundles are in `kaggle_outputs/stage2/clean_v1/`; executed notebooks are in `notebooks/executed/stage2/clean_v1/`. The [failure analysis](results/STAGE2_FAILURE_ANALYSIS.md) is complete: predicted contracts reduce incorrect acceptances but also reject correct programs. The next step is the final write-up and, if pursued, a separately developed method revision followed by an independent benchmark.

To reproduce the completed GPU experiment:

1. Download and extract [the notebook pack](notebooks/kaggle/stage2_six_notebooks.zip).
2. Run [notebook 06](notebooks/kaggle/stage2_06_no_contract_to_generator.ipynb) first, with Internet on and GPU T4 ×2. Use **Save Version → Save & Run All**. Wait for **COMPLETE AND VALIDATED: 100 / 100**.
3. Run notebooks 01–05 in fresh sessions with the same settings, saving each completed bundle.
4. Attach the six outputs to [the CPU collector](notebooks/kaggle/stage2_collect_results.ipynb) and download the final bundle after **600 / 600 verified task records**.

Follow the [complete Kaggle instructions](docs/stage2_six_notebooks.md). The notebooks embed their shared runtime; no new GitHub push, adapter upload, old checkpoint, or separate resume notebook is required. All six supplied Kaggle runs completed without recorded generation or attention failures.

The active next phase follows the [contract-quality improvement plan](docs/contract_quality_improvement_plan.md). A CPU-validated synthetic draft suite and two Kaggle adapter-comparison notebooks and a CPU results collector are ready. Follow the [pilot runbook](docs/contract_quality/RUNBOOK.md) and download the [three-notebook pack](notebooks/kaggle/contract_quality/contract_quality_pilots.zip). GPU pilot results and independent task review are still pending; retraining follows the grounding decision.

The [independent Stage 2 workflow](docs/stage2_independent.md) now runs directly from raw datasets and requests, generating its own enriched-input contracts inside each job. Download the [six-system notebook pack plus collector](notebooks/kaggle/stage2_independent/stage2_independent_notebooks.zip). It requires model weights but no earlier experiment outputs or parser-pilot run. Start with notebook 05 for the proposed system. This new workflow is CPU-tested; its GPU results are pending.

## Setup and local checks

Use Python 3.11 or newer. An existing `.venv/` can be reused.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

[Training instructions](finetune/README.md) cover the separate GPU training dependencies and adapters. Recompute the existing CPU research results with:

```bash
./reproduce_all.sh
```

This script uses the frozen benchmark, downloaded parser/generation inputs, and existing result folders. It does not run the six new GPU evaluations.

## Where to work

- `src/igt/` — contracts, verifier, repair agent, tools, sandbox, and model clients.
- `eval/` — research runners and scoring; `eval/stage2_clean/` implements the current six-notebook workflow.
- `notebooks/kaggle/` — the seven current notebooks, their builder, and the download pack.
- `notebooks/training/` — the parser fine-tuning template.
- `notebooks/executed/stage2/clean_v1/` — completed Stage 2 notebooks, including the collector.
- `benchmark/`, `data/`, `configs/` — frozen tasks, candidate pool, datasets, and experiment settings.
- `finetune/` — training scripts, datasets, configurations, and local trained adapters in `runs/`.
- `kaggle_outputs/` — generation and parser inputs needed for reproducibility.
- `results/` — core scored results, reports, safety evidence, and agent traces.
- `docs/` — specifications and current instructions; `docs/history/` holds earlier plans and workflows.
- `tests/` — regression checks for both current and historical implementations.
- `app/`, `sandbox/`, `tools/` — optional demo, Docker sandbox, and supporting utilities.
- `archive/` — earlier notebooks, superseded runs, optional experiment artifacts, and review material. See the [archive index](archive/README.md).

The layout keeps executable dependencies and frozen research inputs at stable paths. Optional extension source remains available because existing imports and source checks depend on it.

## Research results and references

- [Stage 1 results](results/STAGE1_RESULTS.md)
- [Fine-tuning results](results/FINETUNE_RESULTS.md) and [training/data audit](results/FINETUNE_AUDIT.md)
- [Agent evaluation](results/STAGE2_AGENT_RESULTS.md)
- [Stage 2 development comparisons](results/STAGE2_DEV_RESULTS.md)
- [Stage 2 failure analysis](results/STAGE2_FAILURE_ANALYSIS.md) with [reproducible audit evidence](results/stage2_failure_analysis/audit.json)
- [Completed Stage 2 test metrics](results/stage2_clean_v1/stage2_metrics_test.json) and [summary CSV](results/stage2_clean_v1/stage2_summary.csv)
- [Contract schema](docs/contract_schema.md) and [verdict format](docs/verdict_format.md)
- [Open issues](docs/open_issues.md) and [future work](docs/future_work.md)
- [Mid-submission report](docs/mid_submission_report.md), [PDF](docs/reports/mid_submission_report.pdf)
- [Full historical roadmap](docs/history/implementation_roadmap.md)
- [Project handout](docs/course/handout.txt) and [proposal](docs/course/SSSS_LMA_PROJECT_PROPOSAL.pdf)

Adapters and the Python environment are local, git-ignored assets. Archiving files inside this project organizes the layout without reducing its total disk usage.
