# Stage 2 proposed architecture — fresh implementation

A from-scratch implementation of the proposal's Stage 2 architecture. It imports nothing from `src/`, `eval/` or earlier notebooks. It uses only:

- the two fine-tuned **1.5B parser adapters** (selective facts `p1-v2+sf1`, tool use `parser-tools-v2`);
- **Qwen2.5-3B-Instruct (FP16)** as both the planner/code generator and the repair agent;
- the benchmark tasks (`benchmark/tasks`) and the 4 input CSVs.

The verifier was rewritten from the contract specification (`docs/eval_set_and_contract_guide.md` §5).

```
request ─► INTENT PARSER (1.5B+LoRA) ◄── feedback loop (schema / unknown columns or values / vacuous / uncheckable)
              │ contract (must_change, must_preserve)
              ▼
         PLANNER+GENERATOR (3B) ─► safety (AST) ─► sandbox ─► profiler ─► verifier
                                                                   PASS ─► accept
                                                       VIOLATION/crash ─► REPAIR AGENT (3B, tools, ≤3) ─► re-execute ─► re-verify
                                                             UNCERTAIN ─► withhold (flag)
```

## Notebooks (`notebooks/`)

| Notebook | What differs |
|---|---|
| `01_stage2_proposed_selective_facts.ipynb` | Parser A: deterministic facts (median/mean/mode, per group if requested) inserted into the prompt, one reply, no tools |
| `02_stage2_proposed_tool_use.ipynb` | Parser B: the model calls read-only data tools itself (profile_column, count_rows, inspect_rows, group_profile) and may escalate |
| `03_compare_parsers.ipynb` | CPU: side-by-side metrics, paired outcomes, tasks that differ, per-task drill-down (`show("task_id")`) |

01 and 02 have identical cells except the intro, config and **section 10 (intent parser)**.

## Run on Kaggle

1. Upload these as private datasets:
   - `kaggle_upload/stage2_fresh_data.zip` (tasks and CSVs);
   - `kaggle_outputs/training/selective_facts_v1/selective_facts_1_5b.zip`;
   - `kaggle_upload/tooluse_adapter_1_5b.zip`: the `adapter/` folder only. Don't upload `kaggle_outputs/tool_use_v2/tool_use_train_1_5b.zip` directly: it contains nested ZIPs (`baseline.zip` next to `baseline/`), and Kaggle fails with "Directory already exists: baseline".

   An adapter folder you have already uploaded also works. Each adapter is found by its `igt_prompt_version.txt`.
2. Notebook 01: GPU T4 x2, Internet on, attach the data and selective-facts datasets, then Run All. Notebook 02: same, with the tool-use dataset. Both can run at the same time in separate sessions.
3. Download `stage2_fresh_selective_facts.zip` and `stage2_fresh_tool_use.zip` from `/kaggle/working`, then run notebook 03 with both attached. Suggested storage: `kaggle_outputs/stage2/fresh_v1/`.

Each run: a 2-task smoke run, then 20 dev tasks, then 100 test tasks. It resumes if re-run. Set `MAX_TASKS_PER_SPLIT = 5` for a quick trial.

## Metrics

- **Outcome per task:** correct accepted, incorrect accepted (silent error), or withheld. Withholding is further split by reason, plus "withheld but the final candidate was actually correct" (false rejection).
- **Parser:** feedback rounds, recoveries via feedback, tool attempts and failures, tasks with facts, and contract exact match / F1 vs gold (spec §5.8 keys).
- **Repair:** initial PASS, entered repair, fixed by repair.
- **Cost:** tokens per stage.

## Design choices to know

- The parser feedback loop is new. Schema errors and hard grounding problems withhold the task if the last round still fails. Hard problems are unknown columns, a vacuous contract, or a removal condition that matches no rows. Soft hints, such as a misspelled category value, are sent back once; if they persist, the contract is used anyway.
- The tool parser reuses its trained `Invalid response ... Parser error:` correction channel for this feedback. The selective-facts parser gets a plain follow-up message, since it was trained single-turn.
- Verifier: a problem caused by the output is a VIOLATION, so the repair agent can fix it. Examples are a required column missing, the index renumbered, or an undeclared extra column. Only contract-level problems are UNCERTAIN, and those are withheld without repair. The extra-column check closes the spec's open issue OI-7.
- Evaluator: output equals reference when it has the same row ids, column set and dtype family, and equal values (`atol 1e-6`; categories by order/flag).
- The test set was already used during earlier development, so these runs are diagnostic and not a fresh held-out evaluation.

## Local checks (CPU, no models)

```bash
.venv/bin/python stage2_fresh/tests/test_core.py           # gold contract PASSes its reference, rejects no-op (120/120)
.venv/bin/python stage2_fresh/tests/test_mutations.py      # 413 corrupted outputs, 0 accepted
.venv/bin/python stage2_fresh/tests/test_pipeline_mock.py  # full pipeline with scripted fake LLMs, both parsers
.venv/bin/python stage2_fresh/tests/test_compare_mock.py   # comparison notebook on simulated runs
.venv/bin/python stage2_fresh/build_notebooks.py           # regenerate notebooks + data zip after editing cells/
```

## v2 changes (after the first selective-facts run: 71 / 2 / 27 on test)

**Bugs fixed**
- **Repair agent:** in v1 it wrote a tool call, invented the result and wrote code from it in the same reply (60 of 70 tool calls never received a real result). Generation now stops at `</tool_call>` and the real result is returned before the agent continues.
- **Parser feedback:** the selective-facts adapter (trained single-turn) repeated a byte-identical contract after a follow-up message. Retries are now fresh single-turn prompts with the checker notes before the request: retry 1 is greedy, retry 2 uses seeded sampling (`RETRY_TEMPERATURE`). The tool parser keeps its trained correction channel and samples only after a repeated reply.
- **"mean" as a verb** ("0 and 99 both mean not recorded") was read as a statistic request.
- **Code extraction:** a stray `python` first line, prose or CSV blocks, and cut-off blocks.
- **Verifier crashes became `pipeline_error`:**
  - duplicate output columns;
  - `round`/`clip` on text columns.

  These now become VIOLATION or UNCERTAIN.
- **Repair examples for computed columns** showed `before: None` instead of the source column values.
- **`cast` to string of date-only datetimes** now follows pandas `astype(str)`.
- **Metrics:**
  - "recovered by feedback" counted contracts accepted despite unresolved warnings;
  - feedback rounds counted the final failed check;
  - contract scores sorted order-sensitive `args` and ordered categories, and ignored defaults;
  - contract scores covered only accepted contracts.

- **Second review pass:**
  - multi-level output columns (from `groupby().agg`) crashed the verifier;
  - `list.remove` was wrongly blocked as unsafe;
  - the final parser result came from the last attempt, not the best one, so a broken sampled retry could discard a usable contract;
  - the comparison notebook repeated the "recovered by feedback" miscount.
- **New test suites:** crash fuzzing over 3,000 corrupted contracts and odd outputs (`tests/test_fuzz.py`), and a safety-checker battery of legitimate and dangerous code (`tests/test_safety_legit.py`).

**Checks added**
- **Parser feedback:**
  - the value is in another column;
  - the condition removes every row;
  - the dedup key has no duplicates;
  - duplicate or missing categories;
  - a fill contradicts its `null_count`;
  - facts supplied but unused;
  - JSON parse errors with line and column;
  - unknown columns reported per predicate.
- **Generator/repair:** a short pandas-pattern guide, one free format retry, and a nudge when an attempt reproduces the same failure.
- **Verifier:** the message for a renumbered index (`reset_index` after filtering).

**Validation:**
- No gold contract triggers the new checks.
- None of v1's 71 correct accepts is affected.
- The v1 incorrect accept plus 11 withheld tasks now get targeted feedback.

These fixes were found by inspecting test-set failures, so the v2 test numbers are diagnostic. The 20 dev tasks are the fairer check.
