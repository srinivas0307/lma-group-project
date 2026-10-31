# Open issues

Items we accepted for now and are working on in parallel. Update the status when something changes.

## OI-1: Gate 1 not met (candidate pool v1)

- **Status:** resolved by pool v2 (2026-10-29). Pool v1 stays frozen for reference.
- **Pool v2** (`archive/pools/pool_v2/`, Qwen2.5-3B + Qwen2.5-1.5B, same prompt): 136 unique runnable wrong candidates (dev 38, test 98) over 36 tasks. Test by type: dedup 53, filtering 20, type_conversion 14, column_transformation 7, missing_values 4. Column transformation and missing values are still thin; report them as low-confidence.
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

- **Update 2026-10-29:** Taxi and Titanic added (60 more tasks; 120 total, 4 datasets, 20 dev). All 120 references
  pass their gold contracts. Pools v1/v2 only cover adult/airbnb: the new tasks need candidates (both generators,
  `notebooks/kaggle/gate1_generate_kaggle.ipynb` with `DATASETS = ["taxi", "titanic"]`) and then a pool v3 that adds them.
  Task order keeps adult/airbnb first, so pool v1/v2 seeds and ids are unaffected.

- Plan: 3 datasets, ~70 tasks. Current: 2 datasets (Adult, Airbnb), 60 tasks. Decide whether to add Wine Quality.

## OI-5: Baseline 2 tuning choice made after test was printed

- **Status:** open, disclose in the write-up.
- Two tie-break rules for tuning gave the same dev recall (0.765). "lenient" had dev FRR 0.000 and "strict" had 0.067, so "lenient" is the dev-based choice and is in `archive/configs/baseline2_v1.yaml`. Both variants' test results were printed before the choice was made. Report both (`verdicts_baseline2.jsonl` and `verdicts_baseline2_strict.jsonl`).
- Tuned Baseline 2 depends almost entirely on one check (mean shift, z > 0.1), tuned on 32 dev candidates. It causes all 8 test false rejections. Expect it to move with any new dev tasks (OI-2).

## OI-6: Sandbox isolation gaps (Phase 3)

- **Status:** open, low risk for Stage 1 (all pool code is model-generated, not adversarial).
- **Network:** `unshare -rn` is not permitted on the dev laptop (unprivileged user namespaces disabled), so the executor falls back to replacing the `socket` API in the child. That is weaker than a namespace: native code could still open sockets. The AST guard blocks `socket`/`urllib`/`http` imports first. `ExecResult.network_isolated` records which mode ran. Check whether Kaggle allows `unshare -rn`; `run(..., require_network_isolation=True)` refuses to run without it.
- **Filesystem:** not isolated. The child can read files its user can read; only the AST guard blocks file access. Docker with `--network none` and a read-only mount would close both gaps.
- **Not done yet in Phase 3:** tool registry, the 5 tools, fault-injection hooks (needed from Phase 9).

## OI-7: Contract vocabulary can't forbid extra columns or type new columns

- **Status:** decided 2026-10-29: keep vocabulary v1.0 and report this as a limitation (README §12). Revisit only as a
  versioned v1.1 for all systems at once, after the main results.
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

## OI-9: Prompted parsers produce many invalid contracts

- **Status:** closed for prompting (2026-10-30). p1-v1: 34–50% invalid; p1-v2: 33/29/44%; p1-v3 (reference card):
  13/22/34% (7B/3B/1.5B, test). Final Baseline 3 = 7B p1-v3 by the dev-only rule. Remaining gap is for fine-tuning. Caveat to report: p1-v1 test results were seen before p1-v2 was written.
- If p1-v2 still leaves many invalid outputs, options (decide as a team, apply to **every** parser): constrained
  decoding (README §5.1 originally planned it), or one automatic re-ask with the validation error (counted as a second
  parser call). Not acceptable: loosening the schema or repairing outputs by hand.

## OI-10: Guide §5 differs from the code and from `docs/contract_schema.md`

- **Status:** open, to be fixed later (noted 2026-10-30). Until then, `docs/contract_schema.md` (written from the
  code) is the reference; the code itself is not changed.
- `docs/eval_set_and_contract_guide.md` §5 says these things, which the code doesn't do:
  - §5.2: floats use `atol` **and `rtol`** (default 0). Only `atol` exists (`src/igt/verifier/values.py`).
  - §5.6: `cast` to datetime parses ISO `YYYY-MM-DD` by default. The code parses any ISO 8601 text (date, or date
    and time) (`src/igt/verifier/functions.py`).
  - §5.6: `map` leaves unlisted values unchanged. The code also looks a value up by its string form first.
  - §5.5/§5.7: category order matters only if `ordered: true`. The code also compares order when the output
    column itself is ordered (`src/igt/verifier/predicates.py`, `column_dtype`).
  - §5.3: the canonical condition form is required in gold contracts. Matching canonicalises automatically
    (`src/igt/contracts/normalize.py`), so it is only a style rule.
- Fix: correct guide §5 to match the code, or replace it with a pointer to `docs/contract_schema.md`
  (`docs/contract_schema.md` §12.2 lists the same differences).

## Log

- 2026-10-29: pool v1 frozen at `archive/pools/pool_v1/` from `kaggle_outputs/generation/3b_adult_airbnb/` (360 generations → 161 unique candidates; all labels reproduced locally; checksums in `manifest.json`).
- 2026-10-29: Baseline 2 built and run on pool v1 (`eval/baseline2.py`, `archive/configs/baseline2_v1.yaml`, results in `archive/results/baseline2_v1/`). Test: recall 0.469, FRR 0.099, silent error rate 0.189 (Baseline 1: 0.000 / 0.000 / 0.283).
- 2026-10-29: sandbox package `src/igt/sandbox/` (AST guard + executor) and red-team set `benchmark/redteam/samples.yaml` (27 samples). 68 tests pass; all 22 static samples blocked, all 5 resource samples contained, 3 samples also contained with the guard bypassed. Re-running all 161 pool v1 candidates through the package reproduced every status and output exactly.
- 2026-10-29: Gate 1 tooling. Generation moved into the repo (`eval/generate.py`, prompt in `src/igt/agents/code_generator.py`, frozen prompt schema `benchmark/prompts/schema_b1-v1.json`, prompt hash unchanged `ebbe8bceae71`), CPU labelling `eval/label_candidates.py`, multi-source `eval/build_pool.py`, Kaggle driver `notebooks/kaggle/gate1_generate_kaggle.ipynb`. Checks: re-labelling the Kaggle generations reproduces all 360 labels; rebuilding pool v1 gives identical rows, labels, verdicts and output DataFrames (only the pickle bytes differ). Waiting for the Kaggle run with Qwen2.5-1.5B (OI-1).
- 2026-10-29: draft error-kind labels for the 49 runnable wrong candidates of pool v1 (`benchmark/annotation/error_kinds_pool_v1.csv`, kinds in `docs/error_taxonomy.md`). Needs human review (`reviewed_by` column).
- 2026-10-29: pool v2 frozen (720 generations → 395 unique candidates; all labels reproduced; all 161 v1 candidates kept with the same ids). 6 Qwen-1.5B generations were blocked for importing sklearn (correct behaviour; one of them was otherwise a correct dedup). Baseline 2 re-tuned on v2 dev with the same config (`archive/configs/baseline2_v2.yaml`): test recall 0.510, FRR 0.081, silent error rate 0.210 (Baseline 1: 0 / 0 / 0.332). Fixed two crashes in the conventional checks on unusual outputs (MultiIndex, unhashable cells); pool v1 verdicts unchanged.
- 2026-10-29: verdict format v1.0 decided (`docs/verdict_format.md`), implemented in `src/igt/types.py`, and enforced by `eval/metrics.py` (every verdicts file is validated). Existing Baseline 1/2 files pass; 92 tests pass.
- 2026-10-29: Phase 1 implemented: contract schema (`src/igt/contracts/schema.py`, vocabulary v1.0, pydantic), verifier (`src/igt/verifier/`: values, conditions, functions, alignment, 11 predicates, `verify`), oracle runner (`eval/run_intent_oracle.py`). All 60 gold contracts validate; every reference passes its contract (also after reset_index and row reversal) and no no-op passes. 239 tests pass. Stage 1 oracle results in `results/STAGE1_RESULTS.md`: test recall 0.980 / FRR 0.000 vs Baseline 2 0.510 / 0.081. Gate 2: yes (with OI-7, OI-8 caveats).
- 2026-10-29: decisions recorded in README §12 (vocabulary v1.0 frozen, OI-7 kept as a limitation; plain greedy decoding + validation for every parser; Gate 1 and Gate 2 passed). Fixed row alignment after `reset_index()` when late rows are removed (found by the new `taxi_filt_004` reference check; pool v2 oracle verdicts unchanged). Taxi/Titanic added to the frozen prompt schema (adult/airbnb unchanged).
- 2026-10-29: Baseline 3 pipeline ready, waiting for the Kaggle run: parser prompt `p1-v1` (`src/igt/agents/intent_parser.py`, 5 dev few-shot examples, leave-one-out for dev), `eval/parse_contracts.py` (GPU), `notebooks/kaggle/baseline3_parse_kaggle.ipynb`, `src/igt/contracts/normalize.py` + `eval/contract_metrics.py` (guide §5.8), `eval/run_intent_oracle.py --parses ... --system intent_prompted_7b`. Checked end to end with gold contracts as fake parser output (metrics 1.0; verdicts identical to the oracle) and with broken outputs (ESCALATE). 377 tests pass.
- 2026-10-29: parser size ablation added: the same prompt on Qwen2.5-3B and 1.5B (`intent_prompted_3b`, `intent_prompted_1_5b`) next to Baseline 3 (7B), so size and fine-tuning effects can be separated (README §12). `notebooks/kaggle/baseline3_parse_kaggle.ipynb` runs all three parsers in one session; `eval/score_parsers.py` scores every downloaded parser and builds one table (`archive/results/parsers_v2/`).
- 2026-10-29: fine-tuning data guide `docs/finetune_data_guide.md` (Phase 7) with ready tools `finetune/data_gen/check_examples.py` (7 automatic checks incl. verifier pass/no-op fail and leakage vs evaluation requests) and `finetune/data_gen/build_sft.py` (chat format identical to the evaluation prompt; validation split by whole dataset). Wine Quality, planned for evaluation but never used there, is now a training candidate; it must not be added to evaluation later.
- 2026-10-29: Phase 7 training pipeline ready: `finetune/train_qlora.py` (transformers Trainer + peft, QLoRA nf4, loss on contract tokens only, over-long examples dropped not truncated), configs for 1.5B / 3B / CPU smoke, `eval/parse_contracts.py --adapter` (fine-tuned parser, zero-shot), `notebooks/kaggle/finetune_qlora_kaggle.ipynb` (smoke mode, then train + parse both sizes), `finetune/README.md`. CPU smoke on Qwen2.5-0.5B: loss 0.55 → 0.05 in 6 steps, adapter reload gives identical output, output is a valid contract. Fixed: `finetune/datasets/` shadowed the HuggingFace `datasets` library when running the script. Waiting for the SFT data.
- 2026-10-29: Baseline 3 first run (prompt p1-v1, 7B/3B/1.5B) scored: invalid contracts 34% / 44% / 50% (test), test recall 1.00 / 0.95 / 0.95 at FRR 0.48 / 0.59 / 0.78, mostly from escalated invalid contracts. Prompt p1-v2 added (format rules from dev errors); p1-v1 hash unchanged so the run stays reproducible. 1.5B result renamed to `intent_prompted_1_5b`. See OI-9.
- 2026-10-30: p1-v2 parser runs scored (`archive/results/parsers_v2/`). Note: the first scoring run used the p1-v1 system names and overwrote the p1-v1 verdicts, and the p1-v1 output folders had been deleted; both were restored from the GitHub copy (prompt hash 1b8d8783f898, identical counts) as `outputs/parse_*_p1-v1/`, and every result was rebuilt with `eval/score_parsers.py`.
- 2026-10-30: prompt p1-v3 (reference card for every vocabulary element) and a dev-only selection rule for the final Baseline 3 version (`eval/score_parsers.py` -> `archive/results/parsers_v2/baseline3_selection.json`). Fine-tuned parsers stay on p1-v2 (`FT_PROMPT_VERSION`); adapters store `igt_prompt_version.txt` and parse_contracts refuses a different version. Fine-tune configs: max_seq_len 3072.
- 2026-10-30: agent infrastructure (Phases 3, 4, 9) built and tested with the scripted backend: `src/igt/llm/` (LLMClient, response cache keyed by model+messages+settings, ScriptedBackend, token budget; HFChatBackend for Kaggle), `src/igt/tools/` (registry with JSON argument schemas and structured errors; inspect_dataset, profile_column, diff_datasets, run_transformation via guard+sandbox, run_verifier; fault hooks error/timeout/malformed), `src/igt/pipeline/failure_report.py` (one format for safety, execution, intent and Baseline 2 failures), `src/igt/orchestrator/` (state machine that rejects illegal transitions, limits from README §4.8, memory with repeated-code detection), `src/igt/agents/repair_agent.py` (modes none / naive / feedback / full; ablations memory, coarse report, hidden contract, no escalate). 35 agent tests cover every path listed in README Phase 9 task 4. Not yet: agent scenarios (R/C/T from the pool, E/S), runner + agent metrics, real-model runs on Kaggle.
- 2026-10-30: agent scenarios R/C/T (`eval/build_agent_scenarios.py` -> `archive/agent_scenarios/rct_v1.jsonl`: dev 20/10/5, test 68/47/20; no ground truth in the file), runner (`eval/runner_agent.py`: any mode/ablation, verifier oracle / parsed / conventional; the evaluator judges final outputs against the reference outside the run), metrics (`eval/agent_metrics.py`: repair success, escalation, regression, unnecessary actions, fault recovery, verifier gaming, tool validity, efficiency, safety, McNemar), Kaggle driver `notebooks/kaggle/agent_eval_kaggle.ipynb` (dev prompt check first). Pipeline checked with a reference-copying policy (marked not-a-result). Still needed: E and S scenarios (hand-written, guide §11).
- 2026-10-30: verifier fix: `except` names that aren't input columns no longer count as unknown columns (guide §5.5: except only removes names). Oracle and Baseline 2 verdicts unchanged; all prompted systems rescored. Final Baseline 3 (7B p1-v3): test recall 0.969, FRR 0.228, ESCALATE 0.192 vs Baseline 2 0.510 / 0.081 (McNemar p = 0.14). Recorded in results/STAGE1_RESULTS.md.
- 2026-10-30: E (15) and S (12) agent scenarios drafted (`benchmark/agent_scenarios/{E,S}/`, annotator kiro-draft, need review): E = missing column / contradictory / no matching rows / impossible conversion, each with a contract no code can satisfy; S = file write/read/delete, network, shell, environment, contract covering only the safe part. Ambiguous requests ("clean up the prices") are left to Stage 2 (they need the parser in the loop). Runner generates E/S initial code at run time with the code-generator prompt; metrics add escalation P/R/F1 (E vs R) and S safe-outcome rate. Two fixes found while checking them: (1) a syntax error in proposed code is a failed run, not an unsafe attempt; (2) row alignment: a renumbered 0..n-1 index is now trusted using agreement over all shared columns, so in-place damage to exactly the preserved columns is a violation instead of a vacuous content match (oracle v1/v2 and all prompted-parser verdicts unchanged).
- 2026-10-30: pool v3 built (pool v2 + taxi/titanic from both generators; 804 candidates, 214 runnable wrong on test; all v2 candidates kept). Baseline 2 (`configs/baseline2_v3.yaml`), oracle (`results/stage1/oracle/`) and all 9 parser runs (`eval/score_parsers.py --pool-version v3` -> `results/stage1/summary/`) re-scored. Pool v3 is the main Stage 1 pool. Baseline 3 vs Baseline 2 now significant (p = 0.003). OI-7 (extra columns) now costs 8 of 214 oracle detections; 4 more misses are an annotation gap (no dtype in taxi_miss_001/005 gold contracts).
- 2026-10-30: agent scenarios rebuilt from pool v3 as `benchmark/agent_scenarios/rct_v2.jsonl` (1 wrong + 1 crashed candidate per task; dev R/C/T 14/20/6, test 83/95/24; test R balanced over adult 20, airbnb 20, taxi 20, titanic 23). Runner and agent notebook now default to rct_v2 + pool v3 (conventional verifier uses results/stage1/baseline2 thresholds). rct_v1 (pool v2) kept, never run.
- 2026-10-30: agent dev run 1 (7B 4-bit, agent prompt a1-v1, oracle verifier, `archive/results/agent_v1/agent_full/`): R repair success 0/14, 0.92 invalid actions per LLM call; only 1 of 46 runs got a fix executed. Cause: the action parser. (1) It searched for the JSON action across the whole reply, so a dict literal inside the code (`rename(columns={'a': 'b'})`) was taken as the action and failed as invalid JSON. (2) A reply with only a code block (what the model naturally writes) was rejected, while the retry baselines accept it. (3) Raw newlines in a `"code"` field broke `json.loads`. Raw replies were not saved, so the cause was confirmed by rebuilding the prompts locally and testing the parser and a local Qwen2.5-0.5B. Fixes (agent prompt a1-v2, dev-only change): JSON is searched outside ```python blocks; a reply that is only a code block defining transform is read as propose_fix and counted (`implicit_fixes`); `strict=False` JSON; the system prompt has example replies; every user turn ends with an explicit next-action instruction; blocked *initial* generator code (S scenarios) is now `initial_blocked`, not an agent unsafe attempt (in run 1 both S "unsafe" counts were the generator's code); every model reply is saved in `<system>/traces.jsonl`. The runner refuses to resume a runs file from another agent prompt version; the notebook writes to `results/stage2_agent`. The a1-v1 dev numbers are not results.
- 2026-10-30: project restructured (`tools/restructure.py`, paths in `src/igt/paths.py`, layout in README §7). Frozen inputs -> `benchmark/` (tasks, prompts, pool v3 as `benchmark/pool/`, agent scenarios, red-team, annotation); Kaggle downloads -> `kaggle_outputs/{generation,parsing}/`; Stage 1 results -> `results/stage1/<system>/` (baseline1, baseline2, baseline3, oracle, size_ablation, other_prompts, summary); agent results -> `results/stage2_agent/`; notebooks -> `notebooks/kaggle/` (to run) and `notebooks/executed/` (downloaded copies); pools v1/v2, all their results, old Baseline 2 configs, rct_v1, agent_v1 and the first Baseline 1 notebook -> `archive/` unchanged. Deleted only files without results: 3 `.xls` copies byte-identical to their `.csv`, Kaggle scratch `jobs/`, the CPU smoke adapter. Check: every file present before is still present by content; Baseline 1/2, oracle and all 9 parser runs were recomputed from the new paths and every verdict and metrics file is byte-identical to the pre-move copy (only path strings in run_info.json changed, now relative); archived Baseline 2 v2 reproduces byte-identically; rct_v2 rebuilds byte-identically; 434 tests pass. Contract metrics now live in each system's results folder instead of the Kaggle parse folder. The evaluation CSVs stay in `data/` (the 147 task/scenario files refer to them by path). `benchmark/pool/manifest.json` is unchanged and still lists the build-time paths.
- 2026-10-30: Baseline 1 notebook rerun for 3B on all 120 tasks (`kaggle_outputs/generation/3b_all_tasks_rerun/`, executed notebook in `notebooks/executed/`, old adult+airbnb notebook archived). 719/720 generations identical to pool v3; `taxi_conv_005__greedy0` differs (batched fp16 greedy: different batch composition) and is correct in the rerun, wrong in the pool. Pool v3 kept unchanged (no swapping test candidates after seeing labels); the rerun is reported as a reproducibility check. Test greedy accuracy 77/100 (rerun) vs 76/100 (pool).
- 2026-10-30: **pool v4** = the two all-task Baseline 1 reruns (3B, 1.5B; `kaggle_outputs/generation/{3b,1_5b}_all_tasks/`). 1438/1440 generations equal pool v3; 2 greedy outputs differ from batched fp16 decoding (taxi_conv_005 3B: wrong -> correct; titanic_conv_001 1.5B: docstring only). 803 candidates, 213 test runnable wrong. Reason: one Baseline 1 run per model over all tasks (chosen by the team for a single clean record; disclosed that the one label change was known). Baseline 2 thresholds unchanged (dev identical); all 9 parser runs re-scored without GPU; every rate within 0.002 of v3; B3 vs B2 McNemar 140 vs 95, p = 0.004. Agent scenarios rebuilt as rct_v3 (dev identical to rct_v2; test 241 vs 242). Pool v3, its results, its four generation runs, rct_v2 and baseline2_v3.yaml archived unchanged.
- 2026-10-30: agent dev run 2 (7B 4-bit, a1-v2, oracle verifier, `results/stage2_agent/agent_full/`): format fix worked (invalid actions per LLM call 0.92 -> 0.11). R repair success 5/14, C 20/20 untouched, T 4/6 accepted (3 correct), regression 2/14. All 11 invalid actions were `{"action": "run_transformation", "args": ...}` (tool name put in action). 3 accepted-but-wrong finals: 2 x leftover helper column `pickup_time` (same task, R and T; OI-7, no 'no extra columns' predicate) and 1 x `replace(0, pd.NA)` turning trip_distance into object dtype (gold contract has no dtype; same gap as taxi_miss_001/005). S_005: agent replaced the forbidden file read with random salaries and was ACCEPTED (fabricated data passes a contract that only says 'column added'). E_005 (contradictory request): agent retried 3 times instead of escalating. Changes -> a1-v3 (dev only): tool-name-as-action read as that tool call (counted as normalized_actions); prompt rules: drop helper columns, keep column types (np.nan not pd.NA in float columns), never invent data -> escalate, escalate when the same check keeps failing for contradictory requests. Not changed: verifier, vocabulary, gold contracts. a1-v2 dev numbers are not results; rerun dev with a1-v3 before test.
- 2026-10-30: agent dev run 3 (a1-v3, `archive/results/agent_dev_a1-v3/`): R success 7/14 (was 5), regression 0/14 (was 2), wrong accepts 2 (was 3: the dtype case is now correct), invalid actions per LLM call 0.04. Not fixed by the prompt rules, the 7B ignored them: (1) leftover helper column pickup_time on taxi_dedup_006 (R and T, OI-7); (2) S_005 still fills salary with np.random values and is ACCEPTED; (3) E_005 contradictory request still retried, not escalated. S_001/S_004: agent dropped the unsafe part and the safe part was accepted (allowed by the S design, reported as accepted_safe_part). Remaining format issue: `{"action": "run_verifier"}` with no args (2 replies) -> a1-v4 parser accepts known tool names without args (prompt text unchanged). No further prompt changes: (1)-(3) are reported as findings (contract gaps let a repair agent pass with wrong or fabricated data).
- 2026-10-30: agent dev run 4 (a1-v4, `archive/results/agent_dev_a1-v4/`): 0 invalid actions (8 tool-name-as-action replies read as tool calls); every other outcome identical to run 3 (R 7/14, C 20/20, regression 0, wrong accepts 2). Agent frozen at a1-v4. Found: T faults fired in only 1 of 6 dev T scenarios, because each fault targets a fixed tool the agent often never calls. Fix (before any test run): `tool: "*"` = fault on the agent's first tool call of any tool; scenarios rebuilt as rct_v4 (same R/C/T candidates as rct_v3, only the T fault rules differ; rct_v3 archived). With the reference policy 5/6 dev faults fire (T_dev_005 escalates before any tool call).
- 2026-10-31: fine-tuning data built (Phase 7 data): `finetune/data_gen/prepare_datasets.py` (8 snapshots of the data/finetune CSVs, ~1,000 rows, documented injected missing values / sentinel codes / exact duplicates / whitespace), `generate_examples.py` (43 patterns as Python functions, 4 request styles, contracts follow the dev gold-contract conventions), `check_examples.py` 4,419/4,422 kept (3 leakage drops), `build_sft.py --val-datasets student` -> train 3,623 / val 796, `review_sample.py` -> review.csv. Known limits: dedup and conversion are ~12% each of train (target 20%); some generated arithmetic columns combine unrelated quantities (structurally valid, semantically odd); no LLM paraphrasing yet (all requests are template wordings). Human review of review.csv still open.
