# Intent-Gated Data Transformation Agent

Pandas code generation with an intent-contract parser, deterministic verification, and bounded tool-assisted repair. Team SSSS · LMA Group Project 2026.

## Start here for the report

Use the [report guide](docs/REPORT_GUIDE.md) for the main results, evidence locations, limitations, and which experiments can be compared. It replaces the long chronological list of notebooks as the project entry point.

- [Original six-system Stage 2 evaluation](results/stage2_clean_v1/stage2_summary.csv) and [failure analysis](results/STAGE2_FAILURE_ANALYSIS.md).
- [3B FP16 generator/repair comparison](results/stage2_dual_3b_analysis/ANALYSIS.md): both parser methods give **73 correct, 2 incorrect accepted, 25 withheld** on 100 tasks.
- [7B NF4 repair comparison](results/stage2_dual_7b_analysis/ANALYSIS.md): both methods give **78 correct, 1 incorrect accepted, 21 withheld**. The generator stays 3B FP16.
- [Complete 3B failure inventory](results/stage2_dual_3b_analysis/failure_review/INVENTORY.md) and [improvement plan](docs/stage2_3b_improvement_plan.md).

These experiments use different protocol versions. Previously inspected test cases are diagnostic; do not describe them as a new independent benchmark. The 7B/3B repair comparison changes both model size and precision. See the report guide before combining results.

## Current runnable work

The [current-stack Baseline 2 analysis](results/baselines_hints_v1/BASELINE2_ANALYSIS.md) reports **75/11/14**, versus repair-hints proposed selective **82/2/16** and tools **80/1/19**. Generic distribution checks reject legitimate requested changes; all 100 baseline records passed CPU audit.

The [completed contract-check experiment](results/contract_check_v1/completed/ANALYSIS.md) gives selective facts **76/1/23** and tools **74/1/25**, with lower token use but the shared wrong acceptance still unresolved. All 200 task records pass the CPU audit.

The [combined gold workflow with 7B NF4 repair](docs/stage2_combined_gold_7b.md) is ready for Kaggle. It retains 3B FP16 generation and the same combined procedure/budgets, with an optional comparison against the completed 84/100 combined 3B gold run. GPU results are pending.

The [combined 3B gold-contract diagnostic](docs/stage2_combined_gold.md) is implemented for comparison with both completed parser methods. It needs only its new source ZIP and no parser adapter. Its completed result is **84 correct, zero incorrect accepted, 16 withheld**, versus 74/1/25 for each predicted-contract method; see [the audited gold analysis](results/stage2_combined_gold_analysis/ANALYSIS.md). A post-run collector serialization bug is handled offline.

The [combined 3B end-to-end workflow](docs/stage2_combined_3b.md) is ready for both **selective facts** and **trained tool use**. Each notebook generates fresh contracts and code, then uses the combined repair implementation. Generation and repair remain unquantized 3B FP16; the selected method-specific 1.5B parser adapters are unchanged.

- [Selective-facts Kaggle notebook](notebooks/kaggle/stage2_combined_3b/stage2_combined_3b_selective_proposed_kaggle.ipynb).
- [Tool-use Kaggle notebook](notebooks/kaggle/stage2_combined_3b/stage2_combined_3b_tools_proposed_kaggle.ipynb).
- [Shared source/data ZIP](notebooks/kaggle/stage2_combined_3b/stage2_combined_3b_source.zip).
- [Optional CPU comparison collector](notebooks/kaggle/stage2_combined_3b/stage2_combined_3b_collect_kaggle.ipynb).

Attach the new source dataset and the selected adapter dataset from the corresponding completed 3B run. No GitHub push or previous predictions are needed. Each notebook runs smoke, 20 development tasks and 100 diagnostic test tasks. Completed combined end-to-end results: **74 correct / 1 incorrect accepted / 25 withheld** for each method, compared with original 3B **73/2/25**. Total tokens increase 3.5% for selective facts and 8.4% for tools. See [the analysis](results/stage2_combined_3b_analysis/ANALYSIS.md).

The [completed repair-only development experiment](results/stage2_repair_dev_analysis/v1/ANALYSIS.md) gives control **11/1/5** versus combined **16/0/1** (correct/incorrect accepted/withheld), using 26.1% fewer tokens. This motivates the new integration; it does not establish its full Stage 2 accuracy. Parser correction and information-routing changes remain planned.

## Folder map

- `results/`: report analyses, scores, audits and per-task evidence.
- `docs/`: specifications, current instructions, proposal and submitted report material.
- `kaggle_outputs/`: canonical uploaded runs and training bundles; keep these as original evidence.
- `notebooks/executed/`: executed notebooks backing those runs.
- `notebooks/kaggle/` and `notebooks/colab/`: runnable templates and sealed source packages.
- `src/`, `eval/`, `finetune/`, `tests/`, `benchmark/`, `data/`, `configs/`: implementation, reproducibility inputs and checks.
- `archive/`: earlier experiments and historical evidence.
- `kaggle_uploads/`: builders for optional adapter-containing upload packages. Redundant generated ZIPs were removed; rebuild only when needed.

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_stage2_repair_dev.py tests/agent/test_repair_agent.py
./reproduce_all.sh
```

The reproduction script recomputes existing CPU results; it does not launch new GPU inference. Setup dependencies are in `requirements-dev.txt`; training instructions are in [finetune/README.md](finetune/README.md).

[Uploaded artifact index](docs/uploaded_artifacts.md) · [Cleanup manifest](docs/report_cleanup_manifest_2026-11-15.json) · [Historical index](archive/README.md).
