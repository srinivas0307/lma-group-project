# Future work and open ideas (input for the write-up)

Each idea says what problem it addresses, the evidence for that problem from our own results, and what it would take.
Nothing here is implemented, except where a line says so.

## 1. Letting the parser see the data

The parser gets the column names, dtypes, missing counts and 3 rows, never the whole table (60–140 KB, about
20k–40k tokens per file; real tables can have millions of rows). Some requests need a value only the data can answer.
Evidence (Baseline 3, prompted 7B, p1-v3): on `titanic_miss_004` and `airbnb_miss_004` ("fill with the median") it
wrote the text `"median"` as the fill value; on `titanic_miss_005` (median per class) the contract was invalid; on
`titanic_miss_002` ("most common port") it guessed `"S"`, which happened to be right.

| | Idea | What it would take | Trade-off |
|---|---|---|---|
| **A** | **Tool-using parser:** the parser calls read-only tools (`profile_column` with an optional condition, `count_rows`, `inspect_rows`) before writing the contract | **Built as an extra experiment** (`src/igt/agents/tool_parser.py`, `notebooks/kaggle/tool_parser_kaggle.ipynb`); compared with Baseline 3 on the same 120 tasks | Several model calls per task; the fine-tuned parsers can't use it without training data that contains tool calls |
| **B** | **Richer data summary in the prompt** (deferred, details in §8.3): per column the median / mean for numbers, the most common values for low-cardinality columns, missing counts, a few values with nulls | Change the schema block shared by all prompts; rebuild the fine-tuning data; re-run every parser | One call, no agent loop; but long prompts for wide tables, and it can't answer conditional questions (median *per class*) |
| **C** | **Statistics in the contract vocabulary (v1.1)** (deferred, details in §8.2): e.g. `fill with median(Age)`, `mode(Embarked)`, `median(Age) by Pclass`, computed by the verifier from the input | New relation functions in the schema and verifier; re-check every task; re-run all systems | The cleanest fix for statistics: the contract no longer depends on one snapshot (today a literal like 28.0 is only right for this file, `docs/contract_schema.md` §12.1); but it grows the vocabulary every parser must learn |

A further step is **fine-tuning the small parser to use tools**: generate training examples that contain tool calls and
their results (the data generator already knows which values come from the data), so a 1.5B/3B model learns when to look.

## 2. Closing the known vocabulary gaps (OI-7, OI-8)

- **No predicate forbids extra columns, and new columns are untyped.** Cost: 8 of the oracle's 12 test misses (helper
  columns left in the output), and 3 of the full agent's 4 accepted-but-wrong outputs. Fix: a `no_columns_added`
  (or `columns_exact`) predicate and the convention that every new column gets a `column_dtype`.
- **Annotation gaps** (e.g. `taxi_miss_001/005` don't state `passenger_count`'s dtype): the second annotation study
  (sheets ready in `benchmark/annotation/second_pass/`) is the planned way to find them.
- Both need a versioned v1.1 applied to every system at once, after the main results (README §12). Details: §8.1.

## 3. Refusing and asking instead of completing

- **Unsafe requests.** On the S scenarios the full agent accepted 7 of 9 by doing only the safe part, and in 2 cases
  invented data for the file or web lookup it couldn't use. No unsafe code ran, but no step refuses or flags an unsafe
  *request*: the safety checker blocks code, not intentions. Idea: a request-level check before generation (a small
  classifier or the parser itself marks "needs file / network / shell access"), plus a contract rule that every part of
  the request must be covered.
- **Escalation is rarely the agent's own decision.** It chose `escalate` once in 222 runs; 7 of its 8 escalations on
  impossible requests came from the verifier's UNCERTAIN path. Ideas: an explicit "can this be done?" step before
  repair; escalate after repeated no-progress attempts; train the decision on the E/R scenarios.
- **Ambiguous requests** ("clean up the prices") were left out of the evaluation. A clarifying-question action
  (ask the user, then parse again) would make them testable.

## 4. More reliable contracts

- **Constrained decoding or one re-ask with the validation error** (OI-9). Even the best prompt left 13% (7B) to 34%
  (1.5B) of contracts invalid, and half of Baseline 3's false rejections come from invalid contracts escalating. Must be
  applied to every parser at once to keep the comparison fair.
- **Check the contract against the data before using it:** a `rows_removed_where` that matches no row, or a column
  that doesn't exist, is almost certainly a parsing error. The tool parser (A) can do part of this; a deterministic
  pre-check could re-ask the parser.

## 5. A better repair loop

- About half of all repair attempts made no progress (full agent 51%, feedback 56%, naive 63%;
  `results/STAGE2_AGENT_RESULTS.md`). Ideas: stop or change strategy after a no-progress attempt; show the diff of
  failures between attempts.
- The full agent used `diff_datasets` in only 57% of its repair runs and called `inspect_dataset` more often. A
  stronger hint, or fine-tuning the agent on successful trajectories, could make the diagnosis step routine.
- The agent's gain over plain feedback retry (0.71 vs 0.62) wasn't significant on 82 scenarios, at 5–6× the tokens.
  More scenarios and seeds are needed before claiming tools help repair.

## 6. Scaling to real data (not started; beyond this project)

Today every table is small (891–1,000 rows) and everything runs in pandas in memory: the sandbox gets the whole
table, the verifier aligns both tables row by row, and the tools return summaries computed on the full table. That
breaks on real data in three ways.

### 6.1 Large tables: compute where the data is

- **Problem:** millions of rows don't fit in a prompt (already handled: only summaries go to the model) and may not fit
  in memory for the sandbox and the verifier (not handled). Row alignment (`src/igt/verifier/alignment.py`) and the
  per-row predicates are O(rows) in Python.
- **Idea:** keep the data in a database (DuckDB on Parquet files is the simplest; any SQL engine works) and push the
  work there:
  - **Tools** (`profile_column`, `count_rows`, `inspect_rows`, `diff_datasets`) become SQL queries; only the small
    result goes to the model.
  - **Predicates** become queries over the input and output tables joined on the row id: `rows_removed_where` =
    count of input rows matching the condition that still have a row id in the output; `columns_unchanged` = count
    of joined rows where a column differs; `duplicate_count` = a `GROUP BY` on the key. The verifier then never loads
    the tables into Python.
  - **Generated code** either stays pandas (run on a sample, then on the full data in chunks) or the generator writes
    SQL / DuckDB relational code directly, which the same engine runs.
- **What it would take:** a second backend for the tools and the verifier behind the same interfaces
  (`igt.tools`, `igt.verifier.verify`); a row-id column materialised in the database (the pandas index plays that
  role today); equivalence tests showing the SQL and pandas verifiers give identical verdicts on the current pool.
- **Open questions:** sampling-based checks (verify on a sample, report a confidence) versus exact checks; how to
  keep the sandbox's resource limits when the engine runs outside it.

### 6.2 Wide tables: show only the relevant columns

- **Problem:** a table with hundreds of columns makes the schema block in every prompt long (each column is one line;
  the Taxi file's 20 column lines take about 180 tokens, before the sample rows), and most columns are irrelevant
  to a given request.
- **Idea:** pick the columns the request is about before building the prompt: exact and fuzzy name matching, column
  descriptions, embedding similarity between the request and column names/sample values, or a first cheap model call
  ("which columns does this request touch?"). The rest are summarised in one line ("and 180 other columns, unchanged
  unless named"). The contract can still say `columns_unchanged: "*"`, because `"*"` is expanded against the real
  table by the verifier, not by the model.
- **What it would take:** a column-selection step in `build_messages` (parser and generator), a recall test (does the
  selection keep every column the gold contract names?), and wide-table tasks to measure it on.

### 6.3 Multi-step and multi-table transformations

- **Problem:** the vocabulary describes one step on one table: every predicate compares one input with one output
  through a shared row id. Joins, group-by aggregations, pivots and pipelines of several steps break that.
- **Idea, by case:**
  - **Pipelines** (filter, then fill, then convert): one contract per step, verified on each intermediate output; or
    one end-to-end contract, which is weaker because intermediate damage can cancel out.
  - **Joins:** predicates over row provenance, e.g. "every left row appears exactly once" (left join without
    duplication), "no row of A without a match in B" (inner join), plus key-uniqueness checks on the join key.
  - **Aggregations / pivots:** output rows are groups, not input rows, so row-level predicates don't apply. New
    predicates: one output row per group key, an output column equals `sum/mean/count` of an input column per group.
- **What it would take:** vocabulary v2 (new predicates and a notion of several inputs), a verifier that tracks
  provenance across steps, new tasks and gold contracts, and new parser training data. A separate project rather than
  an extension.

## 7. Stronger evaluation

- **External validity:** a DS-1000 pandas subset (planned as optional); real user requests instead of hand-written ones.
- **More generators:** the pool comes from Qwen2.5-3B and 1.5B; stronger generators make fewer, subtler errors.
- **LLM-as-verifier baseline**, validated against human labels, to show whether a contract adds anything over asking a
  large model "is this output right?".
- **Human labels:** the second annotation of gold contracts, trajectory labels and the review of the 272 error-kind
  drafts are prepared but not done; they turn several "upper bound" numbers into measured ones.
- **Stronger sandbox:** Docker with no network and a read-only filesystem (OI-6); today network isolation on the dev
  machine is a socket stub only.

## 8. Deferred on purpose: changes that require re-running every system

These three would very likely improve the results, but each changes something every system shares (the contract
vocabulary, the gold contracts, or the prompt). Making them now would mean re-running Baselines 2 and 3, the oracle,
the size ablation, the fine-tuning data and training, the agent runs and Stage 2, and changing the vocabulary after
seeing test results would tune the method on the test set (README §12, OI-7). They are recorded here with enough detail
to do them as one versioned change (vocabulary / prompt **v1.1**) after the main results.

### 8.1 Vocabulary v1.1: no extra columns, typed new columns (OI-7)

- **Evidence:** 8 of the oracle's 12 test misses are helper columns left in the output (e.g. taxi dedup code that adds
  `pickup_datetime`); 3 of the full agent's 4 accepted-but-wrong outputs are an extra column or a new column of the
  wrong type (`R_test_050`, `R_test_052`, `R_test_021`). Error-kind breakdown: the oracle catches 9 of 16
  `helper_column_left` candidates.
- **Change:**
  - a predicate `no_columns_added` (or `columns_exact: [...]`): the output has no columns besides the input columns
    and the ones listed in `columns_added`;
  - the convention that every `columns_added` column also gets a `column_dtype` (e.g. `minimum_stay_cost: int`);
  - default-preservation rule 4 of the annotation guide ("no columns are added unless requested") then maps to a
    predicate, as rules 1–3 already do.
- **Where:** `src/igt/contracts/schema.py` (new predicate), `src/igt/verifier/predicates.py`, `normalize.py`
  (matching), the parser prompts (`intent_parser.py`: predicate list and reference card), the fine-tuning templates
  (`finetune/data_gen/generate_examples.py`), and every gold contract in `benchmark/tasks/` (add the predicate; add a
  dtype to new columns). `tests/verifier/test_reference_contracts.py` checks every reference still passes.
- **Re-run:** all Stage 1 systems, the fine-tuning data and training, the agent runs and Stage 2 (each verdict can change).
- **Risk:** stricter contracts raise false rejections when a correct solution keeps a harmless helper column; the answer
  key already counts such outputs as wrong, so the oracle's FRR should stay 0.

### 8.2 Option C: statistics in the contract vocabulary

- **Evidence:** the parser can't compute a statistic from 3 rows (Baseline 3 wrote `"median"` as a fill value on
  `titanic_miss_004` and `airbnb_miss_004`; `titanic_miss_005`, the median per class, was invalid). Gold contracts
  hard-code the value (`28.0`), so they are only right for this snapshot (`docs/contract_schema.md` §12.1).
- **Change:** new relation functions computed by the verifier from the **input** table:
  `median(col)`, `mean(col)`, `mode(col)`, `min/max(col)`, and grouped forms `median(col) by [group cols]`; e.g.
  `{"fn": "const_stat", "args": [{"stat": "median", "col": "Age", "by": ["Pclass"]}]}` in a `column_value_relation`
  with `where: Age is_null`. The verifier evaluates the statistic on the before data (excluding missing values) and
  compares as today.
- **Where:** schema (function list and argument rules), `src/igt/verifier/functions.py`, `normalize.py`, parser prompts
  and reference card, fine-tuning templates (the fill-with-statistic patterns), and the gold contracts that use a
  computed literal (the miss_* tasks with a median / mode; listed by searching the task `notes` for computed values).
- **Re-run:** everything that reads a contract (Stage 1 parsers and oracle, fine-tuning, agent, Stage 2).
- **Risk:** a larger vocabulary for small parsers to learn; ties in `mode`, and float tolerance for means, need fixed
  rules.

### 8.3 Option B: richer data summary in the prompt

- **Evidence:** as for 8.2; also exact category spellings (`Self-emp-inc`), and whether a column has missing values,
  decide how the null rule is written, and 3 sample rows often don't show them.
- **Change:** the schema block shared by the parser and generator prompts adds, per column: the number of distinct
  values; for low-cardinality columns (e.g. ≤ 20 values) all values with counts; for numbers min / median / max; and a
  few example values from rows with missing cells elsewhere.
- **Where:** the frozen prompt schema (`benchmark/prompts/schema_b1-v1.json` -> a new `schema_b2.json`), the user
  templates in `intent_parser.py` and `code_generator.py`, `finetune/data_gen/build_sft.py` (the fine-tuning prompts
  must show the same block), and the live demo's `schema_entry`.
- **Re-run:** every parser (prompted and fine-tuned, after rebuilding the SFT data and retraining) and, if the
  generator prompt changes too, the candidate pool itself, which would invalidate every Stage 1 result. Safer: change
  only the parser prompt and keep the generator on b1-v1.
- **Risk:** long prompts for wide tables (see 6.2); still no answer to conditional questions (a median per class),
  which only tools (A) or statistics in the vocabulary (8.2) handle.

### Order if done

8.1 first (smallest change, clearest measured cost), then 8.2 (removes snapshot-specific literals), then 8.3 only if
the tool-using parser (A) doesn't already close the gap. All three as one v1.1 change, with every result reported in
both versions.
