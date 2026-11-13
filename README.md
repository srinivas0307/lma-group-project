# Intent-Gated Data Transformation Agent

An agent that checks LLM-generated pandas transformations against an intent contract describing what must change and what must be preserved. A bounded repair loop uses tools to inspect failures, propose fixes, and escalate unresolved requests.

**Team SSSS · LMA Group Project 2026**

## Current status and next step

Stage 1 evaluation, parser fine-tuning, agent experiments, and Stage 2 prompt development are complete. The final Stage 2 experiment uses six independent Kaggle evaluation notebooks and a CPU collector. Full completion on Kaggle is still pending.

1. Download and extract [the notebook pack](notebooks/kaggle/stage2_six_notebooks.zip).
2. Run [notebook 06](notebooks/kaggle/stage2_06_no_contract_to_generator.ipynb) first, with Internet on and GPU T4 ×2. Use **Save Version → Save & Run All**. Wait for **COMPLETE AND VALIDATED: 100 / 100**.
3. Run notebooks 01–05 in fresh sessions with the same settings, saving each completed bundle.
4. Attach the six outputs to [the CPU collector](notebooks/kaggle/stage2_collect_results.ipynb) and download the final bundle after **600 / 600 verified task records**.

Follow the [complete Kaggle instructions](docs/stage2_six_notebooks.md). The notebooks embed their shared runtime; no new GitHub push, adapter upload, old checkpoint, or separate resume notebook is required. Real GPU execution remains to be verified on Kaggle.

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
- [Fine-tuning results](results/FINETUNE_RESULTS.md)
- [Agent evaluation](results/STAGE2_AGENT_RESULTS.md)
- [Stage 2 development comparisons](results/STAGE2_DEV_RESULTS.md)
- [Contract schema](docs/contract_schema.md) and [verdict format](docs/verdict_format.md)
- [Open issues](docs/open_issues.md) and [future work](docs/future_work.md)
- [Mid-submission report](docs/mid_submission_report.md), [PDF](docs/reports/mid_submission_report.pdf)
- [Full historical roadmap](docs/history/implementation_roadmap.md)
- [Project handout](docs/course/handout.txt) and [proposal](docs/course/SSSS_LMA_PROJECT_PROPOSAL.pdf)

Adapters and the Python environment are local, git-ignored assets. Archiving files inside this project organizes the layout without reducing its total disk usage.
