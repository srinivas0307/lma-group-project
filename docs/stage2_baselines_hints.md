# Baselines 2/3 and the generator ablation on the current 3B stack

Status: Baseline 2 completed and CPU-audited: **75 correct, 11 incorrect accepted, 14 withheld**, with zero replay mismatches. See [the analysis](../results/baselines_hints_v1/BASELINE2_ANALYSIS.md). No-contract-to-generator/tools is also complete at **78/1/21**, compared with matched proposed tools **80/1/19**; see [the ablation analysis](../results/baselines_hints_v1/NO_CONTRACT_TOOLS_ANALYSIS.md). No-contract/selective is complete at **79/2/19**, compared with proposed selective **82/2/16**; see [the selective ablation analysis](../results/baselines_hints_v1/NO_CONTRACT_SELECTIVE_ANALYSIS.md). Baseline 3 GPU results remain pending.

The six-system comparison (`clean_v1`) used a 7B NF4 repair agent, the a1-v4 repair loop and older parser contracts, so its baseline rows cannot sit next to the current results. This experiment (`stage2-baselines-hints-3b-fp16-v1`) re-runs three of those systems on the current stack: 3B FP16 generator and repair, improved repair loop, contract-check-v1 and repair-hints-v1. Baseline 1 has no repair, but its historical result is not automatically matched to the current generator inputs/runtime: only 67/100 historical direct initial programs match the completed current Baseline 2 initial programs. Run a fresh direct baseline before reporting a clean current six-system comparison. Gold (92/0/8) and proposed (selective 82/2/16, tools 80/1/19) are in `kaggle_outputs/stage2/repair_hints_v1/`.

## System definitions

As in the six-system plan (`eval/runner_stage2.py` PRESETS), the generator sees the contract only in proposed and gold.

| Run | Parser | Contract check | Generator sees | Checker | Repair |
|---|---|---|---|---|---|
| `baseline2` | none | — | request | conventional checks, Stage 1 dev-tuned thresholds (same as clean_v1) | improved loop without contract-derived parts (`repair-no-contract-v1`) |
| `baseline3` | base Qwen2.5-3B, zero-shot p1-v3 prompt, no adapter | yes | request | intent verifier + schema guard | improved loop + hints |
| `no_contract_selective` | fine-tuned 1.5B selective facts | yes | request | intent verifier + schema guard | improved loop + hints |
| `no_contract_tools` | fine-tuned 1.5B tool use | yes | request | intent verifier + schema guard | improved loop + hints |

Baseline 2 keeps the parts of the improved loop that need no contract: failure evidence (check failures, schema delta, row counts), tool-observation labels and one fresh recovery. It drops contract text, contract-aligned examples, the schema guard and hints, which all come from a contract. Its system prompt says the failures come from generic data-quality checks.

Comparisons this enables:

- **Fine-tuning:** `baseline3` vs `no_contract_selective` / `no_contract_tools`. Only the parser differs.
- **Contract to the generator:** `no_contract_*` vs proposed of the same parser. Only the generator prompt differs.
- **Contracts at all:** `baseline2` vs proposed.

## Implementation

`eval/stage2_baselines_hints/` runs the frozen contract-check runner, the one behind the proposed repair-hints runs. It swaps in this experiment's execute step and labels, and restores them afterwards. The source pack reuses the repair-hints pack byte for byte and adds only this package: `notebooks/kaggle/stage2_baselines_hints/stage2_baselines_hints_source.zip`, 304 files. Rebuild it with `.venv/bin/python -m notebooks.kaggle.stage2_baselines_hints.make_notebooks`.

Tests: `tests/test_stage2_baselines_hints.py`. They check:

- the system semantics with scripted models: model load order, request-only generation, no contract or hints in baseline 2, and hints in the contract systems;
- that the thresholds match the Stage 1 file;
- that the sealed pack unpacks and imports correctly.

## Running

Kaggle, Internet on, T4 x2, Save Version → Run All. Each notebook runs a smoke test, 20 dev tasks, then 100 test tasks.

| Notebook | Attach |
|---|---|
| `stage2_baselines_hints_baseline2_kaggle.ipynb` | source zip only |
| `stage2_baselines_hints_baseline3_kaggle.ipynb` | source zip only |
| `stage2_baselines_hints_no_contract_selective_kaggle.ipynb` | source zip + `srinivasu2005/selective-3b-dataset` |
| `stage2_baselines_hints_no_contract_tools_kaggle.ipynb` | source zip + `jagadeeshamudala/tooluse-source` |

Put the ZIPs in `kaggle_outputs/stage2/baselines_hints_v1/`, then run:

    .venv/bin/python tools/analyze_baselines_hints_run.py --run <zip> [--run <zip> ...]

## Limits

The contract check and hints were designed after inspecting these test tasks, so all rows are diagnostic. Baseline 3 uses the 3B model as its prompted parser and the fine-tuned systems use 1.5B. The fine-tuning comparison therefore also changes parser size, and favours baseline 3 on size. Historical clean_v1 rows are shown for reference only; they differ in repair model, loop and parsers.
