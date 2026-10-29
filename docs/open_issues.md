# Open issues

Items we accepted for now and are working on in parallel. Update the status when something changes.

## OI-1: Gate 1 not met (candidate pool v1)

- **Status:** open, accepted for now (2026-10-29). Baseline 2 proceeds on pool v1.
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

## Log

- 2026-10-29: pool v1 frozen at `eval/pool_v1/` from `notebooks/baseline1_outputs/` (360 generations → 161 unique candidates; all labels reproduced locally; checksums in `manifest.json`).
- 2026-10-29: Baseline 2 built and run on pool v1 (`eval/baseline2.py`, `configs/baseline2.yaml`, results in `results/baseline2/`). Test: recall 0.469, FRR 0.099, silent error rate 0.189 (Baseline 1: 0.000 / 0.000 / 0.283).
- 2026-10-29: sandbox package `src/igt/sandbox/` (AST guard + executor) and red-team set `eval/redteam/samples.yaml` (27 samples). 68 tests pass; all 22 static samples blocked, all 5 resource samples contained, 3 samples also contained with the guard bypassed. Re-running all 161 pool v1 candidates through the package reproduced every status and output exactly.
