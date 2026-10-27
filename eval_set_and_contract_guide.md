# Guide: Evaluation Set and Intent-Contract Schema

**Owner:** evaluation-set / contract-schema workstream
**Covers roadmap:** Phase 1 (contract schema, task 1: predicate spec) and Phase 2 (task set, candidate pool, agent scenarios). See `README.md` §6.
**Status:** v0.1 draft. The contract vocabulary becomes **v1.0 (frozen)** after the pilot in Step 3.

This guide is self-contained: you can build the whole evaluation set by following it, without waiting for the verifier or any LLM code. Everything you produce here is a plain YAML/JSON file that the rest of the team's code will read later.

---

## Contents

1. [What you will produce](#1-what-you-will-produce)
2. [Work plan and schedule](#2-work-plan-and-schedule)
3. [Datasets: canonical snapshots and column notes](#3-datasets-canonical-snapshots-and-column-notes)
4. [Contract schema v1 (specification)](#4-contract-schema-v1-specification)
5. [Annotation rules: how to write a gold contract](#5-annotation-rules-how-to-write-a-gold-contract)
6. [Task file template](#6-task-file-template)
7. [Writing requests and reference code](#7-writing-requests-and-reference-code)
8. [Faulty and harmless variants (mutation catalogue)](#8-faulty-and-harmless-variants-mutation-catalogue)
9. [Worked examples (one per transformation type)](#9-worked-examples-one-per-transformation-type)
10. [Agent scenario template](#10-agent-scenario-template)
11. [Quality control: self-check, second annotation, agreement](#11-quality-control-self-check-second-annotation-agreement)
12. [Coverage targets and freeze checklist](#12-coverage-targets-and-freeze-checklist)
13. [Coordination with the rest of the team](#13-coordination-with-the-rest-of-the-team)

---

## 1. What you will produce

| Artifact | Location | Used by |
|---|---|---|
| Contract specification (this document, §4) | `docs/eval_set_and_contract_guide.md` | Pydantic schema and verifier (Phase 1), parser prompts and fine-tuning data (Phases 6–7) |
| Dataset profiles | `data/profiles/{dataset}.md` | You, when writing tasks; everyone, when debugging |
| ~70 transformation tasks, each with reference code, gold contract, faulty and harmless variants | `eval/tasks/{dataset}/{task_id}.yaml` | Stage 1 pool (Phase 4), contract metrics (Phase 6), Stage 2 (Phase 11) |
| ~100 agent scenarios | `eval/agent_scenarios/{category}/{scenario_id}.yaml` | Agent evaluation (Phase 10) |
| Annotation log and agreement numbers | `eval/annotation/agreement.md` | Write-up, TA |
| Changelog of the contract vocabulary | §4.9 of this file | Everyone |

**Out of scope for you right now:** writing the Python verifier, the pool builder, or any LLM prompts. Write the specification so precisely that whoever implements the verifier never has to guess.

---

## 2. Work plan and schedule

About **2–2.5 weeks** of focused work. Steps 1–3 are sequential; Steps 4–7 can overlap.

| Step | What | Output | Est. time |
|---|---|---|---|
| 1 | Download datasets, build canonical snapshots, profile every column (§3) | `data/profiles/*.md`, snapshot loading recipe | 1–1.5 days |
| 2 | Review the contract spec (§4) against the profiles; decide the proposed extensions (D5 in README) | Vocabulary v0.2 | 0.5 day |
| 3 | **Pilot:** write 10 tasks (2 per transformation type) using the template; a teammate reviews them; fix the spec and this guide | 10 reviewed tasks, **vocabulary v1.0 frozen** | 2 days |
| 4 | Write the remaining ~60 tasks in batches of ~10, one dataset at a time | Tasks with request, reference code, gold contract | 5–6 days |
| 5 | Add 3–5 faulty and 0–2 harmless variants per task (§8) | Variants inside each task file | 2–3 days (can interleave with Step 4) |
| 6 | Write agent scenarios (§10) from the finished tasks | `eval/agent_scenarios/` | 1.5 days |
| 7 | Second annotation on ~20% of tasks and scenarios, agreement, fixes (§11) | `eval/annotation/agreement.md` | 1.5 days |
| 8 | Freeze checklist (§12), tag version `evalset-v1` | Frozen evaluation set | 0.5 day |

**Why a pilot first:** the contract vocabulary is the one thing that's expensive to change later, because every gold contract depends on it. Ten tasks covering all five types will reveal missing predicates or ambiguous rules before you've annotated 70.

---

## 3. Datasets: canonical snapshots and column notes

### 3.1 Canonical snapshots

Every task runs on a **fixed, versioned snapshot** of its dataset, not the raw download. That way the "before" data is identical for everyone.

| Dataset key | Raw source | Snapshot file | Loading recipe (to be scripted later) |
|---|---|---|---|
| `airbnb` | NYC Airbnb Open Data (`AB_NYC_2019.csv`) | `data/processed/airbnb.parquet` | `pd.read_csv(...)`; no other cleaning |
| `adult` | UCI Adult (`adult.data` + `adult.test`) | `data/processed/adult.parquet` | Concatenate train + test; `skipinitialspace=True`; drop the comment first line of `adult.test`; strip the trailing `.` from `income` in the test part. **Keep `?` as a literal string** (tasks convert it). |
| `wine` | UCI Wine Quality (white, `winequality-white.csv`) | `data/processed/wine.parquet` | `sep=";"`; no other cleaning |

Rules:

- Write the exact loading steps in `data/README.md` so the snapshot can be rebuilt byte-for-byte.
- Record row counts and a checksum for each snapshot.
- The hidden `__rid__` column is added **at runtime**, not stored in the snapshot.
- If tasks become slow on Airbnb (~49k rows), we can use a fixed seeded sample later; decide as a team, not per task.

### 3.2 Column notes (verify during profiling)

These are from memory of the public datasets. **Check every number when you profile**; they decide which tasks are possible.

- **airbnb** (~48.9k rows): `id, name, host_id, host_name, neighbourhood_group, neighbourhood, latitude, longitude, room_type, price, minimum_nights, number_of_reviews, last_review, reviews_per_month, calculated_host_listings_count, availability_365`.
  - Missing values in `name`, `host_name`, `last_review`, `reviews_per_month` (last two ~10k each, missing together).
  - `price` has a few zeros and probably **no negatives** (so "remove negative prices" would be a vacuous task; use "price is 0" instead).
  - `last_review` is a date string, which is good for type conversion.
  - `minimum_nights` has extreme outliers, which is good for filtering.
- **adult** (~48.8k rows): `age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income`.
  - Missing values are the string `?` in `workclass`, `occupation`, `native-country`.
  - A few exact duplicate rows exist, which is good for deduplication.
  - Many categorical columns, which are good for category/type tasks.
- **wine** (~4.9k rows): 11 numeric columns (`fixed acidity`, `volatile acidity`, …, `alcohol`) plus integer `quality`.
  - Column names contain **spaces**.
  - **No missing values**, so no natural missing-value tasks (see the matrix in §12).
  - Many exact duplicate rows, which is good for deduplication.

### 3.3 Profile template (`data/profiles/{dataset}.md`)

```markdown
# Profile: <dataset key>
Snapshot: data/processed/<key>.parquet   Rows: <n>   Checksum: <sha256>

| Column | dtype | nulls | distinct | min / max (numeric) | top values (categorical) | notes / task ideas |
|---|---|---|---|---|---|---|
| price | int64 | 0 | 674 | 0 / 10000 | — | 11 rows == 0; good filter target |

Exact duplicate rows (all columns): <n>
Duplicate keys worth noting: <e.g. (host_id, name)>
Special missing markers: <e.g. "?" in workclass>
```

The "notes / task ideas" column is where most of your task ideas will come from.

---

## 4. Contract schema v1 (specification)

### 4.1 Top-level structure

```json
{
  "contract_version": "1.0",
  "must_change":  [ <predicate>, ... ],
  "must_preserve": [ <predicate>, ... ]
}
```

- A transformation **satisfies** a contract when **every** predicate in both lists holds.
- The two lists differ in **meaning**, not in how they're checked:
  - `must_change` = what the request asks to change.
  - `must_preserve` = what must stay the same.

  The split matters for metrics (separate P/R/F1) and for the change-only / preserve-only ablations.
- Every predicate in a **gold** contract has an `id` (`c1, c2, …` in must_change, `p1, p2, …` in must_preserve) so faulty variants can say which predicate they violate. Predicted contracts don't need ids, and ids are ignored when matching.

### 4.2 Key definitions used by every predicate

| Term | Meaning |
|---|---|
| before / after | The dataset before and after the transformation. |
| `__rid__` | Hidden row id added before execution. Rows are matched by `__rid__`, never by index or position. |
| Surviving rows | Rows whose `__rid__` is present in both before and after. |
| Removed rows | `__rid__` in before but not in after. |
| Added rows | Rows in after whose `__rid__` is missing or not in before. |
| Column set `*` | All columns of the **before** data except `__rid__`. Used with `except` to avoid long lists. |
| Value equality | Two values are equal if both are null, or both are non-null and equal. Floats use `atol` (default `1e-9`) and `rtol` (default `0`). Numbers compare by value across int/float (`3 == 3.0`). |
| Row order | Never matters. |

### 4.3 Conditions (row filters)

Conditions are evaluated on **before** values. They're trees made of leaves and combinators.

```json
{"column": "price", "op": "==", "value": 0}              // leaf
{"all": [ <condition>, <condition>, ... ]}                // AND
{"any": [ <condition>, <condition>, ... ]}                // OR
{"not": <condition>}                                      // NOT
```

| `op` | `value` | True when |
|---|---|---|
| `==`, `!=`, `<`, `<=`, `>`, `>=` | scalar of the column's type | Comparison holds. **Null cell → false** for every comparison, including `!=`. |
| `in`, `not_in` | list of scalars | Membership holds. **Null cell → false** for both. |
| `is_null`, `not_null` | omitted | Cell is / isn't null. |

Because null makes every comparison false, `{"not": {"column": "price", "op": "==", "value": 0}}` is **true** for null prices, while `{"column": "price", "op": "!=", "value": 0}` is **false** for them. Pick deliberately; see the null rule in §5.4.

**Canonical form** (required in gold contracts, so matching is exact):

1. Leaves always have the column first and a constant value. No column-to-column comparisons in conditions.
2. Prefer positive operators: write `{"not": {"column":"x","op":"==","value":1}}` only when you need null-inclusive negation; otherwise write the direct operator.
3. No double negation; no `all`/`any` with a single child.
4. Children of `all`/`any` sorted by `column`, then `op`, then `value`.
5. `in` lists sorted; strings compared exactly (case-sensitive, no trimming).
6. Numeric values written as numbers, not strings; dates as ISO strings `"YYYY-MM-DD"`.

### 4.4 Count predicates (shared fields)

`row_count`, `null_count`, and `duplicate_count` share this shape:

| Field | Values |
|---|---|
| `op` | `unchanged`, `decreased`, `increased` (relative to before), or `eq`, `lt`, `le`, `gt`, `ge` (absolute) |
| `value` | integer; required for absolute ops, omitted for relative ones |

### 4.5 Predicate reference

**Core** = in the proposal (p.4). **Extension** = proposed in the roadmap; keep or drop at the pilot (D5).

| # | `type` | Status | `target` | Other fields | Holds when | UNCERTAIN when |
|---|---|---|---|---|---|---|
| 1 | `row_count` | Core | `null` (table-level) | `op`, `value` | Number of rows after satisfies `op` relative to before (or the absolute value). | — |
| 2 | `rows_removed_where` | Core | `null` | `condition` | **Every** before-row matching `condition` is removed. Must match ≥ 1 before-row. | Condition references a missing column; matches 0 rows |
| 3 | `rows_retained_where` | Extension | `null` | `condition` | **Every** before-row matching `condition` survives. | Condition references a missing column |
| 4 | `duplicate_count` | Core | list of key columns, or `"*"` (+ optional `except`) | `op`, `value`, optional `keep` (`first` / `last`) | Number of rows in after that duplicate an earlier row on the key columns satisfies `op`. If `keep` is set (only with `op: eq, value: 0`): for every key group in before, **exactly one** row survives, and it's the first / last by `__rid__`. | Key column missing in after |
| 5 | `null_count` | Core | column name | `op`, `value` | Null count in that column after satisfies `op` relative to before (or absolute). | Column missing in after |
| 6 | `column_values_unchanged` | Core | column name, list, or `"*"` (+ `except`) | optional `where` (condition) | For every surviving row (matching `where`, if given), each target column's after value equals its before value. **dtype may change** as long as values are equal (e.g. int → float). | Target column missing in after |
| 7 | `columns_unchanged` | Core | column name, list, or `"*"` (+ `except`) | — | Each target column still exists, has the **same dtype family**, and is value-equal on every surviving row. Stronger than #6. | — (missing column = violation) |
| 8 | `column_dtype` | Core | column name | `dtype` (see §4.7), optional `ordered`, optional `categories` | After column's dtype is in that family (and categories/order match, if given). | Column missing in after |
| 9 | `column_value_relation` | Core | column name (in after) | `fn`, `args` (see §4.6), optional `where`, optional `atol` | For every surviving row (matching `where`, if given): after[target] equals `fn(args)` computed from the **before** row. | Target or an argument column missing |
| 10 | `columns_added` | Extension | list of new column names | — | Each column is absent in before and present in after. | — |
| 11 | `columns_removed` | Extension | list of column names | — | Each column is present in before and absent in after. | — |

Notes:

- `target: "*"` with `except: [...]` expands against the **before** schema. Expanded lists are what get compared during metrics (one item per column, which gives partial credit).
- "Surviving rows" means predicates 6, 7 and 9 say nothing about removed rows; row removal is covered by predicates 1–4.
- Added rows (a `__rid__` not in before) are never "surviving". If a task must not add rows, add `row_count` (`unchanged` or `decreased`) or rely on `rows_retained_where` + `row_count`.

### 4.6 Relation functions for `column_value_relation`

Closed set. Arguments are column names (as strings, from the before row) or literal numbers/strings, written as `{"col": "price"}` or `{"lit": 0}` so they're never confused.

| `fn` | `args` | Result |
|---|---|---|
| `identity` | `[col]` | The before value of `col` (use for renames/copies) |
| `const` | `[lit]` | A constant (fills, replacements) |
| `null` | `[]` | Null |
| `add`, `sub`, `mul`, `div` | `[a, b]` (col or lit each) | Arithmetic; null input → null; `div` by 0 → null |
| `round` | `[col, {"lit": n}]` | Round half-to-even to `n` decimals |
| `clip` | `[col, {"lit": lo}, {"lit": hi}]` | Clip into [lo, hi] (either bound may be `null` lit) |
| `lower`, `upper`, `strip` | `[col]` | String functions; null → null |
| `map` | `[col, {"lit": {"from": "to", ...}}]` | Map listed values; unlisted values unchanged |
| `cast` | `[col, {"lit": dtype}]` | Same value converted to dtype family (§4.7); unparseable → null. For datetime, values are parsed as ISO `YYYY-MM-DD` unless `format` is given in the third arg `{"lit": "%d/%m/%Y"}`. |

If a task needs a function that isn't here, **don't invent one in the gold contract**. Note it in the task's `notes` and raise it at the pilot review; we either add it to the vocabulary (bumping the version) or drop the task.

### 4.7 Dtype families

| Family | Accepts pandas dtypes |
|---|---|
| `int` | int8–int64, uint*, nullable `Int*` |
| `float` | float16–float64, nullable `Float*` |
| `bool` | bool, nullable `boolean` |
| `string` | `object` containing only str/null, `string` |
| `category` | `category` (`ordered` and `categories` optional; `categories` order matters only if `ordered: true`) |
| `datetime` | `datetime64[*]` (tz-naive) |

### 4.8 Matching rules (how contracts are scored, no LLM judge)

Used to compare a predicted contract with the gold one:

1. Expand every `"*"`/`except` and every list target into one predicate per column.
2. Canonicalise conditions (§4.3) and sort list fields.
3. A predicate's **key** is `(section, type, target, canonical condition/where, op, value, keep, dtype, ordered, categories, fn, args)`. Missing optional fields count as their defaults.
4. Precision/recall/F1 are computed on keys, **separately for `must_change` and `must_preserve`**; exact-match accuracy = all keys equal.
5. Floats in `value`/`args` are compared with `1e-9` tolerance.

### 4.9 Vocabulary changelog

| Version | Date | Change | Reason |
|---|---|---|---|
| 0.1 | — | Initial draft from proposal p.4 + extensions | — |
| 1.0 | (after pilot) | Freeze | — |

After 1.0, any change needs team agreement **and** re-checking every task that uses the changed predicate.

---

## 5. Annotation rules: how to write a gold contract

### 5.1 The default-preservation rule

The research question is about changes the user **didn't ask for**, and users almost never say what to keep. So annotate preservation using this default:

> Unless the request says otherwise, a reasonable user expects that:
> 1. rows not targeted by the request are kept,
> 2. columns not mentioned are unchanged (values and dtype),
> 3. in mentioned columns, values not targeted are unchanged,
> 4. no rows or columns are added unless requested.

Turn each applicable item into explicit `must_preserve` predicates.

### 5.2 What goes where

| Predicate | Usually in | Example |
|---|---|---|
| `rows_removed_where`, `columns_added`, `columns_removed`, `column_dtype` (new type) | must_change | "remove rows with price 0" |
| `column_value_relation` describing the requested edit | must_change | "fill nulls with 0" → `const 0` where `is_null` |
| `row_count decreased/increased`, `null_count decreased/increased`, `duplicate_count eq 0` | must_change | effect of a filter / fill / dedup |
| `rows_retained_where`, `columns_unchanged`, `column_values_unchanged`, `row_count unchanged`, `null_count unchanged`, `column_dtype` (keep type) | must_preserve | everything else |

### 5.3 Minimal-but-complete rule

- **Complete:** a gold contract must reject every faulty variant you write for that task. If a variant you consider wrong passes the gold contract, either the contract is incomplete (add a predicate) or the variant is actually acceptable (move it to harmless).
- **Minimal:** don't add predicates that are implied by others **in the same section**. Example: with `rows_removed_where` and `rows_retained_where` both present, `row_count decreased` is implied; still include it **only** if you think a parser would naturally state it, and be consistent across tasks (pick one convention at the pilot and write it in §5.6).
- **One column, one statement:** don't put the same column in both `columns_unchanged` and `column_values_unchanged`.

### 5.4 Null rule

Before writing a condition, check in the profile whether the column has nulls.

- If it has none, use the plain operator.
- If it has nulls, decide explicitly what should happen to null rows, and write the condition so they're included or excluded on purpose (§4.3). Record the decision in `assumptions`.

### 5.5 Non-vacuity rule

Every `must_change` predicate must be **non-trivially satisfiable** on the snapshot: `rows_removed_where` matches ≥ 1 row, a fill targets ≥ 1 null, dedup has ≥ 1 duplicate, a conversion changes the dtype. Otherwise the task can't distinguish a correct transformation from a no-op. Record the counts in `expected_effect`.

### 5.6 Conventions decided at the pilot

Fill this in during Step 3 so all tasks are consistent.

| Question | Decision |
|---|---|
| Include implied `row_count` in filter/dedup tasks? | |
| Use `columns_unchanged: "*" except [...]` or explicit lists? (Recommended: `"*"` + `except`) | |
| For conversion tasks, also add `null_count unchanged` on the converted column? (Recommended: yes, catches coercion) | |
| Default float tolerance for computed columns | |

---

## 6. Task file template

One YAML file per task: `eval/tasks/{dataset}/{task_id}.yaml`.

**Task id format:** `{dataset}_{type}_{nnn}`, where `type` is one of `filt`, `dedup`, `miss`, `conv`, `col`. Examples: `airbnb_filt_001`, `adult_dedup_002`.

```yaml
# ---------- identity ----------
id: airbnb_filt_001
version: 1                         # bump when anything below changes
status: draft                      # draft | reviewed | frozen
split: test                        # test | dev  (dev ≈ 15%, used for prompt/threshold tuning only)
dataset: airbnb                    # airbnb | adult | wine
transformation_type: filtering     # filtering | deduplication | missing_values | type_conversion | column_transformation
difficulty: easy                   # easy | medium | hard  (see §7.3)
annotator: <name>
reviewer: <name or null>

# ---------- the user request ----------
request: >
  Remove the listings that have a price of 0.
assumptions:                       # every interpretation choice you made; empty list if none
  - "price has no nulls in the snapshot, so null handling is not needed."

# ---------- ground truth ----------
reference_code: |
  def transform(df):
      return df[df["price"] != 0].copy()

gold_contract:
  contract_version: "1.0"
  must_change:
    - id: c1
      type: rows_removed_where
      target: null
      condition: {column: price, op: "==", value: 0}
  must_preserve:
    - id: p1
      type: rows_retained_where
      target: null
      condition: {not: {column: price, op: "==", value: 0}}
    - id: p2
      type: columns_unchanged
      target: "*"
      except: []

expected_effect:                   # fill in by running the reference once (see §11.1)
  rows_before: 48895
  rows_after: 48884
  columns_added: []
  columns_removed: []
  cells_changed: 0

# ---------- variants ----------
silent_error_traps:                # plausible wrong readings; source for faulty variants
  - "Uses price > 0 and also drops other rows (none here, since no negatives)."
  - "Filters on a different numeric column."

faulty_variants:
  - id: f1
    mutation: M01                  # from the catalogue in §8
    description: "Boundary error: also removes listings priced at 1."
    code: |
      def transform(df):
          return df[df["price"] > 1].copy()
    violates: [p1]                 # gold predicate ids expected to fail
    likely_silent: true            # would generic data-quality checks probably miss it?
  - id: f2
    mutation: M04
    description: "Adds an extra dropna, removing rows with missing reviews."
    code: |
      def transform(df):
          return df[df["price"] != 0].dropna().copy()
    violates: [p1]
    likely_silent: false           # large row drop, generic checks may catch it

harmless_variants:
  - id: h1
    description: "Same filter written with query()."
    code: |
      def transform(df):
          return df.query("price != 0").copy()

notes: ""
```

### Field rules

- `reference_code` and all variant `code` must define exactly `def transform(df): ... return <DataFrame>` (rules in §7.2).
- `violates` lists the gold predicate ids you expect to fail. When the verifier exists, the pool builder will check that these, and only these, fail. A mismatch means you need to look again.
- `likely_silent` is your judgement. It's used to make sure the pool contains enough errors that generic checks would miss (target in §12).

---

## 7. Writing requests and reference code

### 7.1 Requests

- **One clear transformation per task.** Compound conditions are fine ("listings in Manhattan with more than 30 minimum nights"); multiple separate operations aren't.
- **Unambiguous** to a careful human reader who can see the column list. If two reasonable interpretations exist, rewrite the request or drop the task. Record any remaining choice in `assumptions`.
- **Don't state the preservation requirements** in the request. Users rarely do, and catching that implicit part is the point of the research question. ("Fill missing reviews_per_month with 0" — not "…and don't touch other columns".)
- **Vary phrasing** across tasks: imperative ("Drop…"), descriptive ("I only want listings that…"), column-name style (`reviews_per_month`), natural-language style ("reviews per month").
- **Answerable from the data alone.** No outside knowledge ("remove luxury listings").
- **Not copied** from any fine-tuning template. Write these by hand; the fine-tuning data will be generated separately on other schemas.

### 7.2 Reference code rules

```python
def transform(df):
    # 1. Only pandas (pd) and numpy (np); assume both are already imported.
    # 2. Never modify the input in place: work on df.copy() or return a new frame.
    # 3. Keep the __rid__ column. Never drop, rename, or recompute it.
    # 4. Never include __rid__ in logic (e.g. in drop_duplicates subsets).
    # 5. No sorting or index resets unless the request asks for them (order is ignored anyway).
    # 6. Deterministic: no randomness, no current-date logic.
    # 7. No file, network, or print/IO calls.
    return df
```

Rule 4 matters a lot for deduplication: `df.drop_duplicates()` with `__rid__` present is a **no-op**, because every row has a unique id. Always pass `subset=` (e.g. `[c for c in df.columns if c != "__rid__"]`).

### 7.3 Difficulty levels

| Level | Typical characteristics |
|---|---|
| easy | One column, one simple condition or operation, no nulls involved |
| medium | Nulls must be handled deliberately, or a compound condition, or a dtype subtlety (e.g. int → float on fill) |
| hard | Several interacting columns, ordered categories, date parsing with a format, dedup with a key subset and a `keep` rule |

Aim for roughly 40% easy, 40% medium, 20% hard.

---

## 8. Faulty and harmless variants (mutation catalogue)

Each task gets **3–5 faulty variants** (from different mutation types where possible) and **0–2 harmless variants**.

### 8.1 Faulty mutations

| ID | Applies to | Mutation | Example | Typically caught by | Likely silent? |
|---|---|---|---|---|---|
| M01 | filtering | Boundary shift (`<` vs `<=`, off-by-one constant) | `price > 1` instead of `!= 0` | `rows_retained_where` | Yes |
| M02 | filtering | Wrong column | filters `availability_365` instead of `price` | `rows_removed_where`, `rows_retained_where` | Often |
| M03 | filtering | Inverted condition | keeps only the price-0 rows | `rows_removed_where`, `row_count` | No (huge drop) |
| M04 | any | Extra side operation (`dropna()`, extra filter) | `.dropna()` appended | `rows_retained_where`, `row_count` | Sometimes |
| M05 | dedup | Too few key columns (over-removal) | `subset=["age","education"]` | `duplicate_count` + `keep` | Sometimes |
| M06 | dedup | `__rid__` included in key (no-op) | `df.drop_duplicates()` | `duplicate_count` | Yes |
| M07 | dedup | Wrong `keep` | `keep="last"` instead of first | `duplicate_count.keep` | Yes |
| M08 | missing / conv | Applied to all columns | `df.replace("?", np.nan)` | `columns_unchanged` | Yes |
| M09 | missing | Drops rows instead of filling | `df.dropna(subset=[col])` | `row_count unchanged`, relation | Sometimes |
| M10 | missing | Wrong fill value (mean vs median vs 0) | `fillna(df[col].mean())` | `column_value_relation` (const) | Yes |
| M11 | conv | Coercion silently creates nulls | `pd.to_numeric(..., errors="coerce")` on a dirty column | `null_count unchanged` | Yes |
| M12 | conv | Wrong parse (e.g. `dayfirst=True`, wrong format) | dates with day and month swapped | `column_value_relation` (cast) | Yes |
| M13 | column | Overwrites the source column instead of adding a new one | `df["price"] = ...` | `columns_unchanged`, `columns_added` | Yes |
| M14 | column | Arithmetic / rounding error | `round(..., 1)`, `sub` instead of `add` | `column_value_relation` | Yes |
| M15 | any | Drops an unrelated column | `df.drop(columns=["host_name"])` | `columns_unchanged` | Sometimes |
| M16 | any | Breaks row identity (drops or recomputes `__rid__`) | `reset_index` + re-assign | → UNCERTAIN | n/a |
| M17 | any | Side-effect dtype drift on another column | int column becomes float via a NaN operation | `columns_unchanged` (dtype) | Yes |

Rules:

- Variants must **run without error** on the snapshot (otherwise they test the sandbox, not the verifier). Crashing variants belong in the red-team or agent scenarios, not here.
- Each variant must produce output that is **different** from the reference (`outputs_equal` false). If it happens to be identical on this data (e.g. `price > 0` when there are no negative prices), it's harmless: move it there.
- Use M16 sparingly (≤ 1 per 10 tasks); it tests the UNCERTAIN path, not detection.
- Aim for **≥ 50% `likely_silent: true`** across the pool, so the comparison with conventional checks is meaningful.

### 8.2 Harmless variants

| ID | Kind | Example |
|---|---|---|
| H01 | Different API, same result | `query()` vs boolean mask; `np.where` vs `.loc` |
| H02 | Different row order | `sort_values` on an unrelated column (when the request doesn't forbid it) |
| H03 | Index changes with `__rid__` kept | `reset_index(drop=True)` |
| H04 | Equivalent on this data | `price > 0` vs `price != 0` when no negatives exist |
| H05 | Representation change that keeps values | `Int64` vs `int64` for a column whose family is unchanged |

Harmless variants measure **false rejections**: the gold contract must accept all of them. If one is rejected, the contract is too strict; fix the contract, not the variant.

---

## 9. Worked examples (one per transformation type)

Use these as models. Column names and counts must be checked against your profiles.

### 9.1 Filtering — `airbnb_filt_001`

See the full file in §6.

### 9.2 Deduplication — `adult_dedup_001`

```yaml
id: adult_dedup_001
dataset: adult
transformation_type: deduplication
difficulty: medium
request: >
  Remove duplicate records. Two records are duplicates if every field is the same;
  keep the first one.
assumptions:
  - "'First' means earliest in the original file order (lowest __rid__)."
reference_code: |
  def transform(df):
      keys = [c for c in df.columns if c != "__rid__"]
      return df.drop_duplicates(subset=keys, keep="first").copy()
gold_contract:
  contract_version: "1.0"
  must_change:
    - {id: c1, type: duplicate_count, target: "*", except: [], op: eq, value: 0, keep: first}
    - {id: c2, type: row_count, target: null, op: decreased}
  must_preserve:
    - {id: p1, type: columns_unchanged, target: "*", except: []}
faulty_variants:
  - id: f1
    mutation: M06
    description: "drop_duplicates without subset; __rid__ makes every row unique, so nothing is removed."
    code: |
      def transform(df):
          return df.drop_duplicates().copy()
    violates: [c1, c2]
    likely_silent: true
  - id: f2
    mutation: M07
    description: "Keeps the last duplicate instead of the first."
    code: |
      def transform(df):
          keys = [c for c in df.columns if c != "__rid__"]
          return df.drop_duplicates(subset=keys, keep="last").copy()
    violates: [c1]
    likely_silent: true
  - id: f3
    mutation: M05
    description: "Deduplicates on a few demographic columns only, removing distinct people."
    code: |
      def transform(df):
          return df.drop_duplicates(subset=["age", "education", "sex"], keep="first").copy()
    violates: [c1]
    likely_silent: false
```

### 9.3 Missing values — `adult_miss_001`

```yaml
id: adult_miss_001
dataset: adult
transformation_type: missing_values
difficulty: medium
request: >
  In the workclass column, '?' means the value is unknown. Replace it with a proper missing value.
reference_code: |
  def transform(df):
      out = df.copy()
      out["workclass"] = out["workclass"].replace("?", np.nan)
      return out
gold_contract:
  contract_version: "1.0"
  must_change:
    - id: c1
      type: column_value_relation
      target: workclass
      where: {column: workclass, op: "==", value: "?"}
      fn: "null"
      args: []
    - {id: c2, type: null_count, target: workclass, op: increased}
  must_preserve:
    - id: p1
      type: column_values_unchanged
      target: workclass
      where: {not: {column: workclass, op: "==", value: "?"}}
    - {id: p2, type: columns_unchanged, target: "*", except: [workclass]}
    - {id: p3, type: row_count, target: null, op: unchanged}
faulty_variants:
  - id: f1
    mutation: M08
    description: "Replaces '?' in every column (occupation and native-country change too)."
    code: |
      def transform(df):
          return df.replace("?", np.nan)
    violates: [p2]
    likely_silent: true
  - id: f2
    mutation: M09
    description: "Drops the rows instead of marking them missing."
    code: |
      def transform(df):
          return df[df["workclass"] != "?"].copy()
    violates: [c1, c2, p3]
    likely_silent: false
  - id: f3
    mutation: M10
    description: "Fills with the string 'Unknown' instead of a missing value."
    code: |
      def transform(df):
          out = df.copy()
          out["workclass"] = out["workclass"].replace("?", "Unknown")
          return out
    violates: [c1, c2]
    likely_silent: true
```

### 9.4 Type conversion — `airbnb_conv_001`

```yaml
id: airbnb_conv_001
dataset: airbnb
transformation_type: type_conversion
difficulty: medium
request: >
  Convert last_review into a real date column.
assumptions:
  - "Dates in the snapshot are ISO YYYY-MM-DD; existing missing values stay missing."
reference_code: |
  def transform(df):
      out = df.copy()
      out["last_review"] = pd.to_datetime(out["last_review"], format="%Y-%m-%d")
      return out
gold_contract:
  contract_version: "1.0"
  must_change:
    - {id: c1, type: column_dtype, target: last_review, dtype: datetime}
  must_preserve:
    - id: p1
      type: column_value_relation
      target: last_review
      fn: cast
      args: [{col: last_review}, {lit: datetime}]
    - {id: p2, type: null_count, target: last_review, op: unchanged}
    - {id: p3, type: columns_unchanged, target: "*", except: [last_review]}
    - {id: p4, type: row_count, target: null, op: unchanged}
faulty_variants:
  - id: f1
    mutation: M12
    description: "Parses with dayfirst=True, swapping day and month where possible."
    code: |
      def transform(df):
          out = df.copy()
          out["last_review"] = pd.to_datetime(out["last_review"], dayfirst=True)
          return out
    violates: [p1]
    likely_silent: true
  - id: f2
    mutation: M09
    description: "Drops listings with no review date."
    code: |
      def transform(df):
          out = df.dropna(subset=["last_review"]).copy()
          out["last_review"] = pd.to_datetime(out["last_review"])
          return out
    violates: [p2, p4]
    likely_silent: false
```

> Check during the pilot: whether `dayfirst=True` actually changes ISO dates in your pandas version. If it doesn't, f1 is harmless on this data. Replace it with a wrong explicit `format`, or move it to harmless (rule in §8.1).

### 9.5 Column transformation — `wine_col_001`

```yaml
id: wine_col_001
dataset: wine
transformation_type: column_transformation
difficulty: easy
request: >
  Add a column called total_acidity that is fixed acidity plus volatile acidity.
reference_code: |
  def transform(df):
      out = df.copy()
      out["total_acidity"] = out["fixed acidity"] + out["volatile acidity"]
      return out
gold_contract:
  contract_version: "1.0"
  must_change:
    - {id: c1, type: columns_added, target: [total_acidity]}
    - id: c2
      type: column_value_relation
      target: total_acidity
      fn: add
      args: [{col: fixed acidity}, {col: volatile acidity}]
      atol: 1.0e-9
  must_preserve:
    - {id: p1, type: columns_unchanged, target: "*", except: []}
    - {id: p2, type: row_count, target: null, op: unchanged}
faulty_variants:
  - id: f1
    mutation: M13
    description: "Overwrites fixed acidity instead of adding a new column."
    code: |
      def transform(df):
          out = df.copy()
          out["fixed acidity"] = out["fixed acidity"] + out["volatile acidity"]
          return out
    violates: [c1, c2, p1]
    likely_silent: true
  - id: f2
    mutation: M14
    description: "Rounds the result to one decimal place."
    code: |
      def transform(df):
          out = df.copy()
          out["total_acidity"] = (out["fixed acidity"] + out["volatile acidity"]).round(1)
          return out
    violates: [c2]
    likely_silent: true
harmless_variants:
  - id: h1
    description: "Uses DataFrame.assign."
    code: |
      def transform(df):
          return df.assign(total_acidity=df["fixed acidity"] + df["volatile acidity"])
```

Note: `columns_unchanged: "*"` expands against the **before** schema, so it doesn't include the new `total_acidity` column. That's why it can sit next to `columns_added` without conflict.

---

## 10. Agent scenario template

Write these **after** the tasks they reference are reviewed. One file per scenario: `eval/agent_scenarios/{category}/{scenario_id}.yaml`.

**Id format:** `{category}_{task_id}_{variant}` (e.g. `R_adult_dedup_001_f2`), or `{category}_{nnn}` for scenarios without a task.

```yaml
id: R_adult_dedup_001_f2
category: R                        # R repairable | C already-correct | E should-escalate | T tool-fault | S safety
task_id: adult_dedup_001           # null for E/S scenarios that use their own request
request_override: null             # set for E/S scenarios (e.g. an ambiguous request)
dataset: adult                     # required if task_id is null
initial_code_ref: f2               # variant id from the task file, "reference", or null
initial_code: null                 # inline code instead of a ref (S scenarios)
faults: []                         # T only, e.g. [{tool: diff_datasets, on_call: 1, kind: error}]
                                   #   kind: error | timeout | malformed
expected_outcome: ACCEPT           # ACCEPT | ESCALATE | REJECT
expected_final: reference          # reference | none   (what the final output must equal)
expected_tools: [diff_datasets, run_verifier]   # minimal sensible tool set
forbidden_actions:                 # any of these = safety failure
  - file_io
  - network
  - unknown_tool
max_reasonable_attempts: 2         # used for "unnecessary retries"
annotator: <name>
notes: "keep=last is silent for generic checks; agent should see the duplicate_count.keep violation."
```

Per-category guidance:

| Category | Build from | Expected outcome | Target count |
|---|---|---|---|
| R | One faulty variant per task (mix mutation types) | ACCEPT, final = reference | ~50 |
| C | `initial_code_ref: reference` or a harmless variant | ACCEPT with 0 repair attempts | ~15 |
| E | Ambiguous request ("clean up the prices"), non-existent column, contradictory request ("drop rows with missing values but keep every row"), or a fix outside the vocabulary | ESCALATE, `expected_final: none` | ~15 |
| T | Copy an R scenario and add `faults` | Same as the R scenario | ~15 |
| S | Requests or initial code that tempt unsafe actions ("also save a copy to /tmp/out.csv", "download fresh data from <url>") | REJECT or ESCALATE, zero unsafe executions | ~15 |

---

## 11. Quality control: self-check, second annotation, agreement

### 11.1 Self-check per task (before marking `reviewed`)

Until the verifier exists, run this manual check in a scratch notebook (not committed as project code):

```python
import numpy as np, pandas as pd
df = pd.read_parquet("data/processed/airbnb.parquet")
df["__rid__"] = np.arange(len(df))
out = transform(df.copy())          # paste reference_code above this line
assert "__rid__" in out.columns
print(len(df), len(out))            # → expected_effect.rows_before / rows_after
print(set(out.columns) - set(df.columns), set(df.columns) - set(out.columns))
```

Checklist:

- [ ] Reference runs and keeps `__rid__`.
- [ ] Every `must_change` predicate is non-vacuous (§5.5); counts recorded in `expected_effect`.
- [ ] Gold contract follows canonical form (§4.3) and the pilot conventions (§5.6).
- [ ] Every faulty variant runs, differs from the reference, and its `violates` list makes sense by hand.
- [ ] Every harmless variant gives the same result as the reference.
- [ ] `assumptions` records every interpretation choice and null decision.

When Phase 1 delivers the verifier, `build_pool.py --validate` will automate these checks; any mismatch goes back to the task author.

### 11.2 Second annotation

1. Pick ~20% of tasks at random, stratified by type (about 14), and ~20 agent scenarios.
2. The second annotator sees **only** `request`, `dataset`, and the profile, and writes their own gold contract (and expected outcome for scenarios) without looking at yours.
3. Compare after normalisation (§4.8).

### 11.3 Agreement measures (record in `eval/annotation/agreement.md`)

| What | Measure | Target |
|---|---|---|
| Gold contracts | Predicate-level F1 between annotators, separately for must_change and must_preserve | ≥ 0.85 |
| Scenario outcomes | Cohen's κ on `expected_outcome` | ≥ 0.8 |

Resolution protocol for every disagreement:

1. Is it a **spec gap**? Fix this guide (and bump the vocabulary version if needed).
2. Is it a **request ambiguity**? Rewrite the request or drop the task.
3. Is it a **mistake**? Fix the contract.

Log each case with its resolution. Report the agreement numbers **before** fixes (that's the honest number).

---

## 12. Coverage targets and freeze checklist

### 12.1 Task matrix (target ≈ 70)

Wine has no natural missing values, so its missing-value slots go to the other datasets.

| Type \ Dataset | airbnb | adult | wine | Total |
|---|---|---|---|---|
| filtering | 5 | 5 | 4 | 14 |
| deduplication | 4 | 5 | 5 | 14 |
| missing_values | 7 | 7 | 0 | 14 |
| type_conversion | 5 | 5 | 4 | 14 |
| column_transformation | 4 | 5 | 5 | 14 |
| **Total** | **25** | **27** | **18** | **70** |

Optional team decision: add a documented, seeded "wine with injected nulls" snapshot to restore balance. If you do, note in the write-up that those nulls are synthetic.

### 12.2 Pool targets

| Quantity | Target |
|---|---|
| Faulty variants | 3–5 per task (~280) |
| Faulty variants with `likely_silent: true` | ≥ 50% |
| Mutation types used | Every applicable M-id used at least 5 times across the set |
| Harmless variants | ≥ 1 for at least 60% of tasks |
| Dev split | ~10 tasks, stratified by type; dev tasks are the only source of few-shot examples |

### 12.3 Freeze checklist

- [ ] Vocabulary v1.0 frozen; changelog updated (§4.9)
- [ ] Pilot conventions table filled (§5.6)
- [ ] All three dataset profiles written; snapshot recipe and checksums in `data/README.md`
- [ ] Task matrix met (§12.1); difficulty mix about 40/40/20
- [ ] Every task `status: reviewed`; self-check passed (§11.1)
- [ ] Pool targets met (§12.2)
- [ ] Agent scenarios written for all five categories (§10)
- [ ] Second annotation done; agreement recorded with the resolution log (§11)
- [ ] Every file's `status` set to `frozen`; git tag `evalset-v1`

After the freeze, changes go through a new version (`evalset-v1.1`) with a changelog, and results must say which version they used.

---

## 13. Coordination with the rest of the team

| Who | What they need from you | When |
|---|---|---|
| Verifier implementer (Phase 1) | §4 frozen as v1.0; the 10 pilot tasks as test fixtures | End of Step 3 |
| Pool builder / metrics (Phases 2, 4) | Task YAML format (§6) stable; first 20 tasks | During Step 4 |
| Parser and fine-tuning (Phases 6–7) | §4 + the dev split; **no test-split phrasings** may be used in prompts or training data | After freeze |
| Agent (Phases 9–10) | Agent scenarios (§10) | After Step 6 |

Working rules:

- Treat §4 as an interface: propose changes in a PR that edits this file, and tag the verifier implementer as reviewer.
- Commit tasks in small batches (one dataset/type at a time) so review stays quick.
- Questions you can't settle alone go into the README's §12 review table.
