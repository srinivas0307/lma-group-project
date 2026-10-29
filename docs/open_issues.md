# Open issues

Items we accepted for now and are working on in parallel. Update the status when something changes.

## OI-1: Gate 1 not met (candidate pool v1)

- **Status:** resolved by pool v2 (2026-10-29). Pool v1 stays frozen for reference.
- **Pool v2** (`eval/pool_v2/`, Qwen2.5-3B + Qwen2.5-1.5B, same prompt): 136 unique runnable wrong candidates (dev 38, test 98) over 36 tasks. Test by type: dedup 53, filtering 20, type_conversion 14, column_transformation 7, missing_values 4. Column transformation and missing values are still thin; report them as low-confidence.
- **Source:** Baseline 1 run on Kaggle, 2026-10-28 (`notebooks/new-baseline-1_complete.ipynb`, Qwen2.5-3B-Instruct, greedy + 5 samples at T=0.7).
- **Requirement (README, Gate 1):** roughly 60+ incorrect outputs, spread across transformation types.
- **What we have:** 49 unique runnable wrong candidates (dev 17, test 32). Crashed candidates are not counted, because every verifier rejects them.

  | Type | Dev | Test | Total |
  |---|---|---|---|
  | deduplication | 8 | 22 | 30 |
  | filtering | 7 | 4 | 11 |
  | type_conversion | 2 | 6 | 8 |
  | column_transformation | 0 | 0 | 0 |
  | missing_values | 0 | 0 | 0 |

- **Consequences:** per-type comparisons are only meaningful for deduplication. Column transformation and missing values can't show any verifier difference. Report them as low-confidence.
- **Options (README Gate 1 remedies):**
  1. More samples per task, or a higher temperature.
  2. A second, smaller generator (e.g. Qwen2.5-1.5B-Instruct) with the same prompt.
  3. Harder column_transformation and missing_values tasks with natural pitfalls.
- **Rule:** any new candidates go into a new pool version (`pool_v2`). Never edit `pool_v1` after it is frozen, so results stay comparable.

## OI-2: Small dev split for tuning Baseline 2

- **Status:** open, accepted.
- Dev has 10 tasks (2 per type) and 17 wrong candidates, 0 in two types. Baseline 2 thresholds tuned on it are weakly supported. State this in the write-up. Adding dev tasks would help (ties in with OI-1, option 3).

## OI-3: Notebook fixes for the next Baseline 1 re-run

Not urgent; apply whenever the Kaggle notebook is run again.

- Pin `python-dateutil`, `pytz` and `tzdata` in `PINNED` (pip installed python-dateutil 3.0.0 into the sandbox folder).
- Print crash reasons (`status in CRASH` → `error.value_counts()`). Checked on pool v1: all 16 crashes are the model's own bugs (returns None, `~` on floats, wrong column names, undefined names inside lambdas). None was caused by `keep_definitions` (`top_level_dropped=0`).
- Greedy decoding runs in batches of 8 (left padding, fp16), which can differ slightly from one-at-a-time decoding. Later systems must reuse `generations.jsonl` and not regenerate.
- Pooled Wilson CIs treat 6 samples per task as independent. Use the deduplicated pool and a task-level bootstrap instead (handled in the evaluator, not the notebook).

## OI-4: Scope differences from the plan

- Plan: 3 datasets, ~70 tasks. Current: 2 datasets (Adult, Airbnb), 60 tasks. Decide whether to add Wine Quality.

## OI-5: Baseline 2 tuning choice made after test was printed

- **Status:** open, disclose in the write-up.
- Two tie-break rules for tuning gave the same dev recall (0.765). "lenient" had dev FRR 0.000 and "strict" had 0.067, so "lenient" is the dev-based choice and is in `configs/baseline2.yaml`. Both variants' test results were printed before the choice was made. Report both (`verdicts_baseline2.jsonl` and `verdicts_baseline2_strict.jsonl`).
- Tuned Baseline 2 depends almost entirely on one check (mean shift, z > 0.1), tuned on 32 dev candidates. It causes all 8 test false rejections. Expect it to move with any new dev tasks (OI-2).

## OI-6: Sandbox isolation gaps (Phase 3)

- **Status:** open, low risk for Stage 1 (all pool code is model-generated, not adversarial).
- **Network:** `unshare -rn` is not permitted on the dev laptop (unprivileged user namespaces disabled), so the executor falls back to replacing the `socket` API in the child. That is weaker than a namespace: native code could still open sockets. The AST guard blocks `socket`/`urllib`/`http` imports first. `ExecResult.network_isolated` records which mode ran. Check whether Kaggle allows `unshare -rn`; `run(..., require_network_isolation=True)` refuses to run without it.
- **Filesystem:** not isolated. The child can read files its user can read; only the AST guard blocks file access. Docker with `--network none` and a read-only mount would close both gaps.
- **Not done yet in Phase 3:** tool registry, the 5 tools, fault-injection hooks (needed from Phase 9).

## OI-7: Contract vocabulary can't forbid extra columns or type new columns

- **Status:** open (found 2026-10-29 in the oracle run; gold contracts deliberately not changed).
- The default-preservation rule (guide §6.1, item 4: "no rows or columns are added unless requested") has no
  predicate for columns: `columns_unchanged: "*"` only covers before-columns. Two test errors pass the oracle
  because of this (`adult_miss_005__c07` leaves a helper column; `airbnb_col_001__c03` makes the new column float).
- Options for vocabulary v1.1: a `no_columns_added` predicate (or `columns_exact`), and the convention that new
  columns get a `column_dtype`. Any change needs re-checking all tasks, and results must say which version they use.
  Decide before the parser prompt (Baseline 3) is written, or keep v1.0 and report the gap.

## OI-8: Oracle results depend on unreviewed gold contracts

- **Status:** open. All 60 contracts are `draft` by one annotator; the oracle numbers (recall 0.980, FRR 0.000)
  are an upper bound until the second annotation / agreement study (Phase 2, task 4) is done.
- FRR 0 is expected by construction (correct = equal to the reference, references pass their contracts).

## Log

- 2026-10-29: pool v1 frozen at `eval/pool_v1/` from `notebooks/baseline1_outputs/` (360 generations → 161 unique candidates; all labels reproduced locally; checksums in `manifest.json`).
- 2026-10-29: Baseline 2 built and run on pool v1 (`eval/baseline2.py`, `configs/baseline2.yaml`, results in `results/baseline2/`). Test: recall 0.469, FRR 0.099, silent error rate 0.189 (Baseline 1: 0.000 / 0.000 / 0.283).
- 2026-10-29: sandbox package `src/igt/sandbox/` (AST guard + executor) and red-team set `eval/redteam/samples.yaml` (27 samples). 68 tests pass; all 22 static samples blocked, all 5 resource samples contained, 3 samples also contained with the guard bypassed. Re-running all 161 pool v1 candidates through the package reproduced every status and output exactly.
- 2026-10-29: Gate 1 tooling. Generation moved into the repo (`eval/generate.py`, prompt in `src/igt/agents/code_generator.py`, frozen prompt schema `eval/prompts/schema_b1-v1.json`, prompt hash unchanged `ebbe8bceae71`), CPU labelling `eval/label_candidates.py`, multi-source `eval/build_pool.py`, Kaggle driver `notebooks/gate1_generate_kaggle.ipynb`. Checks: re-labelling the Kaggle generations reproduces all 360 labels; rebuilding pool v1 gives identical rows, labels, verdicts and output DataFrames (only the pickle bytes differ). Waiting for the Kaggle run with Qwen2.5-1.5B (OI-1).
- 2026-10-29: draft error-kind labels for the 49 runnable wrong candidates of pool v1 (`eval/annotation/error_kinds_pool_v1.csv`, kinds in `docs/error_taxonomy.md`). Needs human review (`reviewed_by` column).
- 2026-10-29: pool v2 frozen (720 generations → 395 unique candidates; all labels reproduced; all 161 v1 candidates kept with the same ids). 6 Qwen-1.5B generations were blocked for importing sklearn (correct behaviour; one of them was otherwise a correct dedup). Baseline 2 re-tuned on v2 dev with the same config (`configs/baseline2_v2.yaml`): test recall 0.510, FRR 0.081, silent error rate 0.210 (Baseline 1: 0 / 0 / 0.332). Fixed two crashes in the conventional checks on unusual outputs (MultiIndex, unhashable cells); pool v1 verdicts unchanged.
- 2026-10-29: verdict format v1.0 decided (`docs/verdict_format.md`), implemented in `src/igt/types.py`, and enforced by `eval/metrics.py` (every verdicts file is validated). Existing Baseline 1/2 files pass; 92 tests pass.
- 2026-10-29: Phase 1 implemented: contract schema (`src/igt/contracts/schema.py`, vocabulary v1.0, pydantic), verifier (`src/igt/verifier/`: values, conditions, functions, alignment, 11 predicates, `verify`), oracle runner (`eval/run_intent_oracle.py`). All 60 gold contracts validate; every reference passes its contract (also after reset_index and row reversal) and no no-op passes. 239 tests pass. Stage 1 oracle results in `results/STAGE1_RESULTS.md`: test recall 0.980 / FRR 0.000 vs Baseline 2 0.510 / 0.081. Gate 2: yes (with OI-7, OI-8 caveats).
