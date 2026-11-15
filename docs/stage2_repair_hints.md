# Repair hints for the 3B repair agent (repair-hints-v1)

Status: built and CPU-replayed; GPU runs pending. Phase 2b of the improvement plan. Models stay Qwen2.5-3B FP16 for generation and repair.

## Why

Even with gold contracts, the 3B pipeline gets 84/0/16. With contract checks, selective facts emits 85 reference-compatible contracts but gets only 76 correct. Most remaining losses are the 3B repair agent failing to write code for a contract that is already right. It sees correct failure evidence and then repeats the same pandas mistakes:

- `sort_values` before `drop_duplicates(keep=...)`
- unrequested `dropna`
- discarded `.cat.set_categories(...)` results
- `round()` / `np.round` instead of `Series.round(n)`
- overwriting a preserved column with `pd.to_datetime` to filter
- dropping required columns
- merge-based fills
- `groupby().first()` used for deduplication

## What changes

Only the repair agent's user message (`eval/stage2_repair_hints/agent.py`). After the existing evidence it adds:

1. **Recipes** (`recipes.py`): pandas lines for the predicates that failed (all `must_change` predicates if the candidate crashed), derived only from the predicted contract and the input columns. They follow verifier semantics: conditions and values come from the original input rows, null cells never match, keep first/last refers to input order, and rounding is `Series.round`.
2. **Lints** (`lint.py`): static AST checks for the pitfalls above. A lint is reported only when the contract makes it relevant.

Everything else is the same frozen code, executed unchanged: parser, contract-check correction, generator, prompts, budgets, schema guard, verifier and fresh recovery. `eval/stage2_repair_hints/runner.py` swaps only the repair function and run labels, and restores them afterwards. Code that passes on the first try never sees a hint.

## CPU replay (no model calls)

`tools/replay_repair_hints.py` writes [results/repair_hints_v1/replay.json](../results/repair_hints_v1/replay.json).

- **Recipe semantics.** All 120 dev/test gold contracts compile (`recipes.assemble`) to a program that passes the guarded verifier and equals the reference output (120/120; this is also a unit test).
- **Lint coverage on gold-run failures.** A relevant lint fires on 14 of the 16 failures, usually on every attempt. The remaining two get recipes only:
  - `adult_dedup_003`: helper column; the exact output-column list is shown.
  - `titanic_filt_005`: invalid boolean indexing.
- **False hints.** No lint fires on initially correct gold-run code. In the checked runs, the dedup-groupby lint fires on three correct candidates that use `groupby().first()` on null-free data. Those candidates passed, so the hint would never be shown to them.
- **Prompt cost.** The hint block is about 140–180 tokens on average, at most about 375, against a 32k context.

The replay cannot show whether the 3B model follows the hints. That needs the GPU runs.

### What contract-following code would expose

Compiling the **predicted** contracts of the checked runs shows the ceiling and the risk:

| | Withheld with a contract | Compiles to the correct output | Compiles to a wrong output |
|---|---|---|---|
| selective | 13 | 9 | 3 (+1 fails verifier) |
| tools | 17 | 11 | 5 (+1 fails verifier) |

Better contract-following turns wrong contracts into wrong accepts. In both checked runs, every task whose contract kept an unresolved contract-check issue after the correction turn had a bad contract. This includes the shared wrong accept `airbnb_filt_002`. No task with a correct final result had one.

`tools/analyze_repair_hints_run.py` therefore also reports an **offline admission gate**: withhold an accepted result when check issues remain unresolved. Applied to the existing selective checked run it gives 76/0/24 instead of 76/1/23. The gate was designed after seeing these test tasks, so it is diagnostic only.

## Running it

1. `.venv/bin/python -m notebooks.kaggle.stage2_repair_hints.make_notebooks`. Already built in `notebooks/kaggle/stage2_repair_hints/`: source pack `stage2_repair_hints_source.zip`, manifest `277f9e13…`, 297 files.
2. On Kaggle, Internet on, T4 x2, Save Version → Run All. Each notebook runs a smoke test, 20 dev tasks, then 100 test tasks.

   | Notebook | Attach | Baseline |
   |---|---|---|
   | `stage2_repair_hints_gold_kaggle.ipynb` | the source zip only | 84/0/16 |
   | `stage2_repair_hints_selective_kaggle.ipynb` | source zip + `srinivasu2005/selective-3b-dataset` | 76/1/23 |
   | `stage2_repair_hints_tools_kaggle.ipynb` | source zip + `jagadeeshamudala/tooluse-source` | 74/1/25 |

3. `.venv/bin/python tools/analyze_repair_hints_run.py --pipeline <gold|selective|tools> --run <zip>`. It reports paired gains and losses, new wrong accepts, initial-code reproduction (it should be complete, since only repair changed), the hints and lints shown, and the gated variant.

Tests: `tests/test_stage2_repair_hints.py`.

## Limits

The hints were designed after inspecting failures on these test tasks, so gains are diagnostic, not independent evidence. Recipes describe the contract, not the request: a wrong contract gets a confidently wrong recipe. Constraint-only predicates (for example `null_count` with no fill rule) get a note, not code.
