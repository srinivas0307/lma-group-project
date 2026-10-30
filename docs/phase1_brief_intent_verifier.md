# Brief: build the intent verifier and produce the `intent_oracle` results (Phase 1 + MVP-1)

> **Status (2026-10-29): implemented** — `src/igt/contracts/`, `src/igt/verifier/`, `eval/run_intent_oracle.py`,
> tests in `tests/verifier/`. Results in `archive/results/intent_oracle_v2/`. The steps below describe what was built
> and are the reference for reviewing or changing it.

## In one sentence

Write the code that **checks a candidate's output against a task's gold contract** and says PASS / VIOLATION /
UNCERTAIN, then run it on every candidate in pool v2 and write `verdicts_intent_oracle.jsonl`.

## What already exists (don't redo it)

| Thing | Where | Notes |
|---|---|---|
| Gold contracts, 60 tasks | `benchmark/tasks/*/*.yaml`, field `gold_contract` | The **input** to your code. Still `status: draft`. |
| Contract specification | `docs/eval_set_and_contract_guide.md` **§5** | Exact meaning of every predicate, condition, function and dtype family. **Your code implements §5.** |
| Candidate pool | `archive/pools/pool_v2/pool_v2.jsonl` (395 candidates) | One row per candidate: `candidate_id`, `task_id`, `split`, `ran_ok`, `output_file`, ... |
| Candidate outputs | `archive/pools/pool_v2/outputs/<candidate_id>.pkl` | The DataFrame each candidate produced (only those that ran). Load with `pickle`. |
| Input data | `data/adult_dev.csv`, `data/airbnb_dev.csv` | Load with `pd.read_csv`; the default index 0..n-1 is the row id. |
| Verdict format + helpers | `docs/verdict_format.md`, `src/igt/types.py` | `Verdict.combine`, `predicate_failure`, `uncertain_failure`, `execution_failure`, `verdict_record`. |
| Metrics | `eval/metrics.py` | Validates your file and builds the comparison table. |

An analogy: the gold contract is the **marking scheme**, your verifier is the **examiner**, the verdicts are the
**marked answer sheets**. The marking scheme exists; nobody marks the papers yet.

## What to build (files from README §7)

```
src/igt/contracts/schema.py        load + validate a contract (pydantic), expand "*" / except
src/igt/verifier/alignment.py      match after-rows to before-rows (index, then content fallback)
src/igt/verifier/conditions.py     evaluate a condition tree on the before data (§5.3)
src/igt/verifier/predicates/*.py   one check per predicate type (§5.5)
src/igt/verifier/intent_verifier.py  run all predicates of a contract -> Verdict
eval/run_intent_oracle.py          loop over the pool -> verdicts_intent_oracle.jsonl
tests/verifier/                    tests per predicate + the reference check below
```

### Step 1. Contract schema (`schema.py`)
- Pydantic models for the structure in §5.1, the conditions in §5.3, the count fields in §5.4,
  and the 11 predicate types in §5.5.
- Validation: unknown predicate type or field → error; a target column that doesn't exist in the
  before data → the verifier returns `UNCERTAIN` with reason `unknown_column`.
- Expand `target: "*"` with `except: [...]` against the **before** columns.
- Freeze the names now ("vocabulary v1.0", §5.9): Baseline 3's parser prompt is written against them.

### Step 2. Row alignment (`alignment.py`) — §5.2
- **Normal case:** match after rows to before rows by index label.
- **Lost index** (labels not unique, not a subset of the before index, or renumbered 0..m-1 after rows
  were removed): match by the content of shared columns.
- If rows still can't be matched one-to-one → `UNCERTAIN`, reason `rows_unmatched`.
- Output: surviving rows (before id → after id), removed rows, added rows, and which method was used.

### Step 3. Conditions (`conditions.py`) — §5.3
- Leaves `{column, op, value}`, combinators `all` / `any` / `not`.
- **Null cell → false for every comparison, including `!=`, `in`, `not_in`.**
- Evaluated on the **before** data.

### Step 4. Predicates — §5.5 (start with the most used)

| Order | Predicate | Used in contracts | Holds when (short) |
|---|---|---|---|
| 1 | `columns_unchanged` | 60 | each target column still exists, same dtype family, same values on surviving rows |
| 2 | `row_count` | 60 | row count after vs before satisfies `op` (`unchanged`, `decreased`, `eq`, ...) |
| 3 | `column_value_relation` | 25 | after[target] == fn(args) from the before row, on surviving rows (`where` optional); fns in §5.6 |
| 4 | `column_values_unchanged` | 19 | values equal on surviving rows; dtype may change |
| 5 | `null_count` | 19 | null count in the column satisfies `op` |
| 6 | `rows_removed_where` | 15 | every before row matching `condition` is gone (must match ≥1 row, else UNCERTAIN) |
| 7 | `rows_retained_where` | 15 | every before row matching `condition` survives |
| 8 | `column_dtype` | 14 | dtype family (§5.7), plus `ordered` / `categories` if given |
| 9 | `duplicate_count` | 12 | no duplicates on the key; with `keep`, exactly one row per key group survives and it is the first/last by index |
| 10 | `columns_added` | 7 | listed columns absent before, present after |
| 11 | `columns_removed` | 2 | listed columns present before, absent after |

Functions actually used by `column_value_relation`: `const` (9), `identity` (3), `map` (3), `div`, `clip`,
`cast` (2 each), `sub`, `mul`, `lower`, `strip` (1 each). Value equality: both null, or equal; floats
`atol=1e-9` unless the predicate gives `atol`; `3 == 3.0`.

Each predicate returns: holds / violated (+ expected, observed, up to 5 evidence row ids) / uncertain (+ reason).

### Step 5. Verifier (`intent_verifier.py`)
```python
from igt.types import Verdict, predicate_failure, uncertain_failure

def verify(contract, before, after) -> Verdict:
    failures = []
    for direction in ("must_change", "must_preserve"):
        for pred in contract[direction]:                  # check ALL predicates, don't stop at the first
            r = check(pred, before, after, alignment)
            if r.violated:
                failures.append(predicate_failure(pred.id, pred.type, direction, pred.target,
                                                  r.expected, r.observed, r.detail, r.evidence_rows))
            elif r.uncertain:
                failures.append(uncertain_failure(r.reason, r.detail, pred.id))
    return Verdict.combine(failures)                      # any violation -> VIOLATION, else uncertain -> UNCERTAIN
```
Add an option to use only `must_change` or only `must_preserve` (needed for the ablations).

### Step 6. Runner (`eval/run_intent_oracle.py`)
For each candidate in `archive/pools/pool_v2/pool_v2.jsonl`:
- `ran_ok == false` → `Verdict.combine([execution_failure(c["status"], c["error"])])`
- else load the input CSV and `outputs/<candidate_id>.pkl`, read the task's `gold_contract`, call `verify`.
- Write `verdict_record("intent_oracle", "intent", c, verdict)` lines to
  `archive/results/intent_oracle_v2/verdicts_intent_oracle.jsonl`.

**Never** read `reference_code`, the reference output or `is_correct` inside the verifier or runner.

## How to know it's right (tests)

1. **Per predicate:** a pass case, a violation case, and edge cases (empty output, all-null column,
   missing column → UNCERTAIN, reset index → content matching, unmatchable rows → UNCERTAIN).
2. **Reference check (the most important test):** for all 60 tasks, run the task's `reference_code`
   (with `igt.sandbox.safe_run`) and verify it against its own gold contract. **Every reference must PASS.**
   Also: returning the input unchanged must **not** pass. Any failure here is either a verifier bug or a
   mistake in the gold contract; fix it and write it down.
3. **Format check:** `eval/metrics.py` must accept your file (it validates every line).

## Done when

```
.venv/bin/python -m pytest -q                          # all tests pass, including the 60 reference checks
.venv/bin/python eval/run_intent_oracle.py             # writes the verdicts file
.venv/bin/python eval/metrics.py --pool archive/pools/pool_v2/pool_v2.jsonl \
   --verdicts archive/pools/pool_v2/verdicts_baseline1.jsonl archive/results/baseline2_v2/verdicts_baseline2.jsonl \
              archive/results/intent_oracle_v2/verdicts_intent_oracle.jsonl
```
The last command prints the main table (README Gate 2): does `intent_oracle` catch more wrong candidates
than `baseline2` (test recall 0.510) at a similar false-rejection rate (0.081)?

## Example to check by hand

`adult_dedup_003` ("remove repeated fnlwgt values and keep the first record for each"), candidate
`adult_dedup_003__c00` sorts by `fnlwgt` before `drop_duplicates(keep='first')`, so for 7 groups a later row is
kept. Expected verdict: `VIOLATION` on `c1` (`duplicate_count` on `fnlwgt` with `keep: first`), evidence rows
96, 147, 365, ... Baseline 2 accepted this candidate (row count and distributions look normal).

## Who does what (suggestion)

- **Friend:** steps 1–5 (schema, alignment, conditions, predicates, verifier) and their tests.
- **Srinivas:** step 6 (runner) and the reference check, using the existing pool, sandbox and metrics;
  meanwhile Baseline 3 part 1 (parser prompt and Kaggle generation).
- Agree the vocabulary freeze (step 1) first, because Baseline 3 depends on it.
