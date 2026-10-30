# Intent-Contract Schema, vocabulary v1.0 (reference)

This is the full reference for the contract format: every field, every predicate, every relation
function, the validation rules, and exactly how the verifier checks a contract.

| | |
|---|---|
| Vocabulary version | `1.0` (frozen 2026-10-29) |
| Schema (validation) | `src/igt/contracts/schema.py` (`load_contract`) |
| Verifier (checking) | `src/igt/verifier/` (`verify`) |
| Matching predicted vs gold | `src/igt/contracts/normalize.py`, `eval/contract_metrics.py` |
| Verdict format | `docs/verdict_format.md` |
| Annotation rules | `docs/eval_set_and_contract_guide.md` §6 |

Where this file and the guide's §5 disagree, this file follows the code. The differences are listed in §12.

## Contents

1. [Top-level structure](#1-top-level-structure)
2. [Core concepts](#2-core-concepts)
3. [Conditions](#3-conditions)
4. [Count operators](#4-count-operators)
5. [Predicates](#5-predicates)
6. [Relation functions](#6-relation-functions-for-column_value_relation)
7. [Dtype families](#7-dtype-families)
8. [Validation rules](#8-validation-rules-load-time)
9. [How verification works](#9-how-verification-works)
10. [Matching a predicted contract with the gold one](#10-matching-a-predicted-contract-with-the-gold-one)
11. [Patterns per transformation type](#11-patterns-per-transformation-type)
12. [Known limitations and differences from the guide](#12-known-limitations-and-differences-from-the-guide)

---

## 1. Top-level structure

```yaml
contract_version: "1.0"
must_change:            # what the request asks to change
  - <predicate>
must_preserve:          # what must stay the same
  - <predicate>
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `contract_version` | string | no | `"1.0"` | Must be exactly `"1.0"`. |
| `must_change` | list of predicates | no | `[]` | |
| `must_preserve` | list of predicates | no | `[]` | |

Rules:

- A transformation **satisfies** a contract when every predicate in both lists holds.
- Both lists are checked the same way. The split only matters for metrics (scored separately) and for the
  change-only / preserve-only ablations.
- Unknown fields anywhere in the contract are rejected (strict schema).
- Every predicate may have an `id`. Gold contracts use `c1, c2, …` in `must_change` and `p1, p2, …` in
  `must_preserve`. A predicate without an `id` gets one by position. Ids are ignored when matching
  predicted with gold contracts.

---

## 2. Core concepts

| Term | Meaning |
|---|---|
| **before / after** | The DataFrame given to `transform(df)` and the one it returns. |
| **Row id** | The **index** of the before data (0..n−1 as loaded from CSV). No helper column is added. |
| **Alignment** | Matching each after row to the before row it came from (see §9.2). |
| **Surviving rows** | Before rows matched to an after row. |
| **Removed rows** | Before rows with no match in after. |
| **Added rows** | After rows with no match in before. They are never "surviving". |
| **Column set `"*"`** | All columns of the **before** data. Combine with `except: [...]` to leave some out. |
| **Key columns** | Columns the contract says are unchanged: targets of `columns_unchanged`, and of `column_values_unchanged` *without* `where`. Used to match rows by content when the index was lost. |
| **Null** | `None`, `NaN`, `NaT`, `pd.NA`. Two nulls are equal to each other. |
| **Row order** | Never matters. |

### 2.1 Value equality

Used by every predicate that compares cells.

| Case | Equal when |
|---|---|
| Either value is null | Both are null. |
| Either value is a boolean | Both are booleans with the same value (`True` ≠ `1`). |
| Both are numbers (int or float) | `abs(a − b) <= atol`, default `atol = 1e-9`; `3 == 3.0`; infinities only equal themselves. |
| Both are datetimes | Same timestamp. |
| One is a string, the other isn't | Never equal (`"1"` ≠ `1`). |
| Otherwise | `a == b` (strings exactly: case-sensitive, no trimming). |

Category cells are compared by their values, not their codes.

---

## 3. Conditions

Conditions select rows. They are always evaluated on the **before** data.

### 3.1 Grammar

A condition is exactly **one** of these four shapes:

```yaml
{column: price, op: "==", value: 0}          # leaf
{all: [<condition>, <condition>, ...]}       # AND, at least 1 child
{any: [<condition>, <condition>, ...]}       # OR, at least 1 child
{not: <condition>}                           # NOT
```

### 3.2 Leaf operators

| `op` | `value` | True when |
|---|---|---|
| `==`, `!=` | scalar (not a list or mapping) | Cell equals / doesn't equal the value. |
| `<`, `<=`, `>`, `>=` | scalar | Comparison holds. |
| `in`, `not_in` | list | Cell is / isn't in the list. |
| `is_null`, `not_null` | must be omitted | Cell is / isn't null. |

### 3.3 Null rule

**A null cell makes every comparison false**, including `!=`, `in` and `not_in`. Only `is_null` and
`not_null` look at nulls.

| Condition | Null cell |
|---|---|
| `{column: Age, op: "!=", value: 0}` | false |
| `{not: {column: Age, op: "==", value: 0}}` | **true** |
| `{column: Embarked, op: not_in, value: [S, C]}` | false |

To include null rows on purpose, add an `is_null` branch:

```yaml
any:
  - {column: Embarked, op: not_in, value: [S, C]}
  - {column: Embarked, op: is_null}
```

### 3.4 Type rules

- `==`, `!=`, `in`, `not_in`: a string never equals a non-string. Numbers compare across int/float.
- `<`, `<=`, `>`, `>=` use Python comparison. Strings compare lexicographically, which orders ISO date
  text (`"2026-01-01 00:00:00"`) the same as time.
- If the before column is a datetime dtype and the value is a string, the value is parsed as a timestamp.
- A comparison Python can't do (for example string `<` number) makes the predicate **UNCERTAIN**
  (`predicate_not_evaluable`).
- A condition column that doesn't exist in before makes the predicate **UNCERTAIN** (`unknown_column`).

### 3.5 Writing style

Matching canonicalises conditions automatically (§10), so the order of children and of `in` lists
doesn't matter for scoring. For readability, gold contracts follow these conventions:

1. Column first, constant value. No column-to-column comparisons (not supported).
2. Use the direct operator; use `not` only for null-inclusive negation.
3. Numbers as numbers (`0`, not `"0"`).

---

## 4. Count operators

Shared by `row_count`, `null_count` and `duplicate_count`.

| `op` | Kind | `value` | Holds when (after count `a`, before count `b`) |
|---|---|---|---|
| `unchanged` | relative | must be omitted | `a == b` |
| `decreased` | relative | must be omitted | `a < b` |
| `increased` | relative | must be omitted | `a > b` |
| `eq` | absolute | integer, required | `a == value` |
| `lt` / `le` | absolute | integer, required | `a < value` / `a <= value` |
| `gt` / `ge` | absolute | integer, required | `a > value` / `a >= value` |

Prefer relative operators where they say enough, so the contract doesn't depend on the row count of a
particular file. `eq 0` is fine because it doesn't depend on the data either.

---

## 5. Predicates

Eleven predicate types. **Core** ones are from the proposal; **extension** ones were added and are part
of v1.0.

### 5.0 Summary

| # | `type` | Status | `target` | Extra fields | Usually in | Row-level¹ |
|---|---|---|---|---|---|---|
| 1 | `row_count` | core | `null` | `op`, `value` | either | no |
| 2 | `rows_removed_where` | core | `null` | `condition` | must_change | yes |
| 3 | `rows_retained_where` | extension | `null` | `condition` | must_preserve | yes |
| 4 | `duplicate_count` | core | column / list / `"*"` | `except`, `op`, `value`, `keep` | must_change | yes |
| 5 | `null_count` | core | one column | `op`, `value` | either | no |
| 6 | `column_values_unchanged` | core | column / list / `"*"` | `except`, `where` | must_preserve | yes |
| 7 | `columns_unchanged` | core | column / list / `"*"` | `except` | must_preserve | yes |
| 8 | `column_dtype` | core | one column | `dtype`, `ordered`, `categories` | either | no |
| 9 | `column_value_relation` | core | one column (in after) | `fn`, `args`, `where`, `atol` | either | yes |
| 10 | `columns_added` | extension | list of columns | — | must_change | no |
| 11 | `columns_removed` | extension | list of columns | — | must_change | no |

¹ Row-level predicates need the alignment. If alignment is ambiguous they return **UNCERTAIN**
(`rows_unmatched`) instead of being checked.

For every predicate, a referenced column that doesn't exist in before gives **UNCERTAIN**
(`unknown_column`) before the check runs (§9.1). The "Uncertain" lines below list only the extra cases.

Usage in the 120 current tasks: `row_count` 120, `columns_unchanged` 120, `column_value_relation` 57,
`column_values_unchanged` 37, `null_count` 35, `column_dtype` 32, `rows_removed_where` 29,
`rows_retained_where` 29, `duplicate_count` 24, `columns_added` 12, `columns_removed` 4.

Each subsection below gives the fields, when the predicate holds, when it is violated or uncertain, and
an example.

---

### 5.1 `row_count`

The number of rows after, compared with before.

| Field | Type | Required |
|---|---|---|
| `type` | `"row_count"` | yes |
| `target` | `null` | no (must be null) |
| `op`, `value` | §4 | `op` yes |

- **Holds:** `len(after)` satisfies `op` relative to `len(before)` (or the absolute `value`).
- **Violated:** otherwise.
- **Uncertain:** never.

```yaml
- {id: c2, type: row_count, target: null, op: decreased}     # filter / dedup / drop
- {id: p4, type: row_count, target: null, op: unchanged}     # fills, conversions, new columns
```

---

### 5.2 `rows_removed_where`

Every before row matching the condition must be gone.

| Field | Type | Required |
|---|---|---|
| `type` | `"rows_removed_where"` | yes |
| `target` | `null` | no |
| `condition` | condition (§3) | yes |

- **Holds:** none of the matching before rows survive.
- **Violated:** at least one matching row survives. Evidence = those row ids.
- **Uncertain:** the condition matches **0 rows** in before (the task is vacuous on this data), or it
  can't be evaluated (§3.4).

```yaml
- id: c1
  type: rows_removed_where
  target: null
  condition: {column: Fare, op: "==", value: 0}
```

---

### 5.3 `rows_retained_where`

Every before row matching the condition must survive. This is the mirror of `rows_removed_where` and
catches over-deletion.

| Field | Type | Required |
|---|---|---|
| `type` | `"rows_retained_where"` | yes |
| `target` | `null` | no |
| `condition` | condition (§3) | yes |

- **Holds:** every matching before row survives (0 matching rows also holds).
- **Violated:** at least one matching row was removed. Evidence = those row ids.
- **Uncertain:** the condition can't be evaluated.

```yaml
- id: p1
  type: rows_retained_where
  target: null
  condition:
    any:
      - {column: Age, op: ">=", value: 18}
      - {column: Age, op: is_null}          # unknown ages must stay
```

---

### 5.4 `duplicate_count`

Duplicates on a set of key columns, optionally with a rule for which row survives.

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | `"duplicate_count"` | yes | |
| `target` | column, list, or `"*"` | yes | Key columns. |
| `except` | list of columns | no | Removed from the key (useful with `"*"`). |
| `op`, `value` | §4 | `op` yes | |
| `keep` | `first` / `last` | no | Only allowed with `op: eq, value: 0`. |

How duplicates are counted: a row is a duplicate if an **earlier** row has the same values on all key
columns. Nulls count as equal to each other; floats are compared after rounding to 9 decimals.

- **Holds (without `keep`):** the after duplicate count satisfies `op` relative to the before count.
- **Holds (with `keep`):** the count is 0 **and** every key group in before keeps **exactly one** row,
  which is its first (lowest index) or last (highest index) row. If rows were matched by content, any row
  identical to that one in every column is accepted.
- **Violated:** count wrong (evidence = duplicate rows), or a group keeps zero, two, or the wrong row.
- **Uncertain:** a key column is missing in after.

```yaml
- {id: c1, type: duplicate_count, target: [Ticket], op: eq, value: 0, keep: first}
- {id: c1, type: duplicate_count, target: "*", except: [fnlwgt], op: eq, value: 0, keep: last}
```

---

### 5.5 `null_count`

The number of nulls in one column.

| Field | Type | Required |
|---|---|---|
| `type` | `"null_count"` | yes |
| `target` | column name | yes |
| `op`, `value` | §4 | `op` yes |

- **Holds:** after null count satisfies `op` relative to the before null count (or absolute `value`).
- **Violated:** otherwise.
- **Uncertain:** the column is missing in after.

```yaml
- {id: c2, type: null_count, target: Cabin, op: eq, value: 0}            # a fill removed every null
- {id: p2, type: null_count, target: last_review, op: unchanged}         # conversion didn't coerce
- {id: c2, type: null_count, target: passenger_count, op: increased}     # sentinel -> missing
```

---

### 5.6 `column_values_unchanged`

Values stay the same on surviving rows. The **dtype may change** (for example int → float or str →
category) as long as the values are equal.

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | `"column_values_unchanged"` | yes | |
| `target` | column, list, or `"*"` | yes | |
| `except` | list of columns | no | |
| `where` | condition | no | Only rows matching it (on before) are checked. |

- **Holds:** on every surviving row (matching `where`), each target's after value equals its before value.
- **Violated:** some cells differ. Evidence = those row ids.
- **Uncertain:** a target column is missing in after, or `where` can't be evaluated.

```yaml
- {id: p1, type: column_values_unchanged, target: Sex}                     # conversion keeps values
- id: p1
  type: column_values_unchanged
  target: Age
  where: {column: Age, op: not_null}                                        # fill only touches nulls
```

---

### 5.7 `columns_unchanged`

The strong form: each column still exists, has the **same dtype family**, and every surviving row has
the same value. This is the standard "nothing else changed" predicate.

| Field | Type | Required |
|---|---|---|
| `type` | `"columns_unchanged"` | yes |
| `target` | column, list, or `"*"` | yes |
| `except` | list of columns | no |

- **Holds:** all of the above.
- **Violated:** a column is missing (renamed or dropped), its dtype family changed, or values differ.
- **Uncertain:** never.

```yaml
- {id: p2, type: columns_unchanged, target: "*", except: []}          # filters, dedup, new columns
- {id: p3, type: columns_unchanged, target: "*", except: [Age]}       # everything except the edited column
```

Don't put the same column in both `columns_unchanged` and `column_values_unchanged`.

---

### 5.8 `column_dtype`

The after column has a given dtype family (§7).

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | `"column_dtype"` | yes | |
| `target` | column name | yes | |
| `dtype` | `int`, `float`, `bool`, `string`, `category`, `datetime` | yes | |
| `ordered` | bool | no | Only with `dtype: category`. |
| `categories` | list | no | Only with `dtype: category`. |

- **Holds:** the after column's family equals `dtype`. For categories, when given:
  - `ordered` must equal the column's ordered flag;
  - `categories` must equal the column's categories. Order matters if either the contract or the column
    is ordered; otherwise they are compared as sets.
- **Violated:** otherwise.
- **Uncertain:** the column is missing in after.

```yaml
- {id: c1, type: column_dtype, target: last_review, dtype: datetime}
- {id: c1, type: column_dtype, target: Pclass, dtype: category, ordered: true, categories: [3, 2, 1]}
```

---

### 5.9 `column_value_relation`

The after value of a column is a function of the **before** row. This describes the requested edit
itself (fills, derived columns, mappings, casts).

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | `"column_value_relation"` | yes | |
| `target` | column name | yes | May be a new column (doesn't need to exist in before). |
| `fn` | one of §6 | yes | |
| `args` | list of `{col: name}` / `{lit: value}` | depends on `fn` | |
| `where` | condition | no | Only rows matching it (on before) are checked. |
| `atol` | float | no | Numeric tolerance; default `1e-9`. |

- **Holds:** on every surviving row (matching `where`), `after[target] == fn(args)` using the value
  equality of §2.1.
- **Violated:** some rows differ. The detail shows the first example (`expected …, got …`).
- **Uncertain:** the target is missing in after, `where` can't be evaluated, or the function raises.

```yaml
- id: c1
  type: column_value_relation
  target: Cabin
  where: {column: Cabin, op: is_null}
  fn: const
  args: [{lit: Unknown}]

- id: c2
  type: column_value_relation
  target: fare_per_mile
  fn: div
  args: [{col: fare_amount}, {col: trip_distance}]
  atol: 1.0e-9
```

---

### 5.10 `columns_added`

| Field | Type | Required |
|---|---|---|
| `type` | `"columns_added"` | yes |
| `target` | list of column names | yes |

- **Holds:** every listed column is in after.
- **Violated:** a listed column is missing in after.
- **Uncertain:** a listed column already exists in before.

```yaml
- {id: c1, type: columns_added, target: [num_relatives]}
```

---

### 5.11 `columns_removed`

| Field | Type | Required |
|---|---|---|
| `type` | `"columns_removed"` | yes |
| `target` | list of column names | yes |

- **Holds:** none of the listed columns is in after.
- **Violated:** a listed column is still there.
- **Uncertain:** a listed column doesn't exist in before (`unknown_column`).

```yaml
- {id: c2, type: columns_removed, target: [Pclass]}
```

A **rename** is `columns_added` (new name) + `columns_removed` (old name) + `column_value_relation`
with `fn: identity` from the old column.

---

## 6. Relation functions (for `column_value_relation`)

Closed set. Arguments are written `{col: name}` (value from the before row) or `{lit: value}` (a
constant), so the two are never confused.

| `fn` | `args` | Result for one row | Null handling |
|---|---|---|---|
| `identity` | `[col]` | The before value of `col` (renames, copies). | null → null |
| `const` | `[lit]` | The constant. | — |
| `null` | `[]` | Null. | — |
| `add` | `[a, b]` col or lit | `a + b` | any null → null |
| `sub` | `[a, b]` | `a − b` | any null → null |
| `mul` | `[a, b]` | `a × b` | any null → null |
| `div` | `[a, b]` | `a / b`; **`b == 0` → null** | any null → null |
| `round` | `[col, lit n]` | `numpy.round(value, n)` (half to even) | null → null |
| `clip` | `[col, lit lo, lit hi]` | Value clipped into `[lo, hi]`; either bound may be `{lit: null}` (no bound). | null → null |
| `lower` / `upper` / `strip` | `[col]` | String function; non-string values unchanged. | null → null |
| `map` | `[col, lit mapping]` | Mapped value. A value is looked up as-is, then as its string form; unlisted values stay unchanged. | null → null |
| `cast` | `[col, lit dtype]` or `[col, lit dtype, lit format]` | The same value converted to the dtype family (rules below); unparseable → null. | null → null |
| *any type error* | | null | |

`cast` rules:

| Target dtype | Result |
|---|---|
| `datetime` | `pd.to_datetime(value, format=…)`; default format is ISO 8601 (date, or date and time). |
| `int` | Number if it is a whole number, else null. |
| `float` | Number as float. |
| `string` | Strings unchanged; other values `str(value)` (so `1 → "1"`, `1.0 → "1.0"`). |
| `bool` | Strings: `true/1/yes → True`, `false/0/no → False` (case-insensitive, trimmed), else null. Other values: `bool(value)`. |
| `category` | Value unchanged (pair with `column_dtype` to check the dtype). |

Argument rules (checked at load time, §8):

| `fn` | Number of args | First arg must be a column | Literal-only positions |
|---|---|---|---|
| `identity` | 1 | yes | — |
| `const` | 1 | — | 1 |
| `null` | 0 | — | — |
| `add`, `sub`, `mul`, `div` | 2 | — | — |
| `round` | 2 | yes | 2 |
| `clip` | 3 | yes | 2, 3 |
| `lower`, `upper`, `strip` | 1 | yes | — |
| `map` | 2 | yes | 2 (must be a mapping) |
| `cast` | 2 or 3 | yes | 2 (a dtype family), 3 |

Not expressible in v1.0: nested expressions (`a + b + 1`), column-to-column conditions, datetime
arithmetic, string extraction (regex), group statistics. Use literals computed from the data (for
example a median) and say so in the task `notes`.

---

## 7. Dtype families

| Family | pandas dtypes |
|---|---|
| `int` | `int8`–`int64`, `uint*`, nullable `Int*` |
| `float` | `float16`–`float64`, nullable `Float*` |
| `bool` | `bool`, nullable `boolean` |
| `string` | `string`, or `object` whose non-null values are all `str` |
| `category` | `category` |
| `datetime` | `datetime64[*]` without a time zone (any unit) |

Anything else gets its own family name and matches no contract dtype. This includes `object` with mixed
values (family `object`) and time-zone-aware datetimes (family `datetime_tz`).

---

## 8. Validation rules (load time)

`load_contract(dict)` raises `ContractError` (shown as UNCERTAIN `invalid_contract` by the verifier) when:

| Rule | Applies to |
|---|---|
| `contract_version` is not `"1.0"` | contract |
| An unknown field is present | everywhere |
| `type` is not one of the 11 predicate types | predicate |
| A condition isn't exactly one of leaf / `all` / `any` / `not` | condition |
| A leaf lacks `column` or `op` | condition |
| `is_null` / `not_null` has a `value` | condition |
| `in` / `not_in` value isn't a list | condition |
| A comparison value is missing, a list, or a mapping | condition |
| `all` / `any` is empty | condition |
| A relative count op has a `value`, or an absolute one lacks it | `row_count`, `null_count`, `duplicate_count` |
| `keep` is used without `op: eq, value: 0` | `duplicate_count` |
| `ordered` / `categories` used with a dtype other than `category` | `column_dtype` |
| Wrong number of args, first arg not a column, or a literal-only position holds a column | `column_value_relation` (§6) |
| `map` second arg isn't a mapping; `cast` dtype isn't a family | `column_value_relation` |

A JSON Schema of the format (usable for constrained decoding) comes from `Contract.model_json_schema()`.

---

## 9. How verification works

```python
from igt.verifier import verify
verdict, info = verify(contract, before_df, after_df)     # info["alignment"] = "index" | "content"
```

It is deterministic, never sees the reference solution, and checks every predicate (no early stop).

### 9.1 Steps

1. **Load the contract.** If it is invalid → `UNCERTAIN (invalid_contract)`.
2. **Check the output.** If it isn't a DataFrame → `UNCERTAIN (predicate_not_evaluable)`. Duplicate
   column names in after: only the first is kept.
3. **Align rows** (§9.2).
4. **For each predicate:**
   - a column it refers to is not in before → `UNCERTAIN (unknown_column)`;
   - it is row-level and the alignment is ambiguous → `UNCERTAIN (rows_unmatched)`;
   - otherwise run the check → holds / violated / uncertain.
5. **Combine:** any violation → `VIOLATION`; otherwise any uncertain → `UNCERTAIN`; otherwise `PASS`.

### 9.2 Row alignment

**Index alignment** is used when the after index can be trusted:

- the labels are unique, have the same number of levels as before, and are all before labels; and
- if the labels are exactly `0, 1, …, m−1` (they might be a `reset_index`), the rows must agree with the
  before rows those labels point to. A row agrees when more than half of its key columns are equal.
  - Same number of rows as before: at least **90%** of rows must agree. This tolerates damaged values
    while still catching a sort followed by `reset_index`.
  - Fewer rows than before: **every** row must agree. A reset after removing rows shifts every later
    row, even if only a few rows at the end are affected.

Labels that are not `0..m−1` (they have gaps or are out of order) are always trusted.

**Content alignment** is used otherwise:

- Rows are matched on the key columns (§2), or on all shared columns if the contract has none.
- Before rows identical in **every** column are interchangeable; any of them may be picked.
- If after rows match several before rows that differ elsewhere, and only some of them survived, the
  alignment is **ambiguous**. Row-level predicates then return `UNCERTAIN (rows_unmatched)`.

### 9.3 Outcomes

| Status | Meaning | Outcome (Stage 1) | Outcome (repair on) |
|---|---|---|---|
| `PASS` | Every predicate holds | `ACCEPT` | `ACCEPT` |
| `VIOLATION` | At least one predicate is violated | `REJECT` | `REPAIR` |
| `UNCERTAIN` | Nothing violated, something undecidable | `ESCALATE` | `ESCALATE` |

`UNCERTAIN` reasons: `invalid_contract`, `unknown_column`, `predicate_not_evaluable`, `rows_unmatched`.

### 9.4 Failure entries

Every violated or uncertain predicate adds one entry to `verdict.failures` (full format in
`docs/verdict_format.md` §3):

```json
{"kind": "predicate", "predicate_id": "c1", "predicate_type": "duplicate_count",
 "direction": "must_change", "target": ["host_id"], "expected": "eq 0", "observed": 182,
 "detail": "182 rows still share a host_id", "evidence_rows": [12, 87, 301]}

{"kind": "uncertain", "reason": "rows_unmatched", "detail": "...", "predicate_id": "p2"}
```

Evidence rows are **before** row ids, at most 5 per predicate.

---

## 10. Matching a predicted contract with the gold one

Used to score intent parsers (`eval/contract_metrics.py`). There is no LLM judge.

1. **Expand** `"*"` / `except` and list targets of `duplicate_count`, `column_values_unchanged`,
   `columns_unchanged`, `columns_added` and `columns_removed` into one key per column (partial credit).
2. **Canonicalise conditions:**
   - single-child `all` / `any` are unwrapped;
   - nested `all` inside `all` (or `any` inside `any`) are flattened;
   - double negation is removed;
   - children and `in` lists are sorted;
   - whole-number floats become ints, other floats are rounded to 9 decimals.
3. **Key** = (section, type, target column, condition / where, op, value, keep, dtype, ordered,
   categories, fn, args). Missing optional fields count as their defaults. `id` and `atol` are not part
   of the key. Unordered `categories` are compared as a sorted set.
4. **Score:** precision / recall / F1 on keys, separately for `must_change` and `must_preserve`.
   Exact match = all keys equal in both sections. An invalid prediction counts as empty.

---

## 11. Patterns per transformation type

These are the conventions used in all 120 current tasks (draft, pending the pilot decision D5).

| Type | must_change | must_preserve |
|---|---|---|
| Filtering | `rows_removed_where` (cond) + `row_count decreased` | `rows_retained_where` (complement, nulls decided explicitly) + `columns_unchanged "*"` |
| Deduplication | `duplicate_count eq 0` with `keep` + `row_count decreased` | `columns_unchanged "*"` |
| Missing values: fill | `column_value_relation const` where `is_null` + `null_count eq 0` | `column_values_unchanged` where `not_null` + `columns_unchanged "*" except [col]` + `row_count unchanged` |
| Missing values: drop | `rows_removed_where is_null` + `row_count decreased` | `rows_retained_where not_null` + `columns_unchanged "*"` |
| Missing values: sentinel → null | `column_value_relation null` where `== sentinel` + `null_count increased` | `column_values_unchanged` where `!= sentinel` + `columns_unchanged "*" except [col]` + `row_count unchanged` |
| Type conversion (values kept) | `column_dtype` | `column_values_unchanged` + `null_count unchanged` + `columns_unchanged "*" except [col]` + `row_count unchanged` |
| Type conversion (values re-encoded) | `column_dtype` | `column_value_relation cast` (or `map`) + `columns_unchanged "*" except [col]` + `row_count unchanged` |
| New column | `columns_added` + `column_value_relation` | `columns_unchanged "*"` + `row_count unchanged` |
| Edit a column in place | `column_value_relation` | `columns_unchanged "*" except [col]` + `row_count unchanged` (+ `column_dtype` if the type must stay) |
| Rename | `columns_added` + `columns_removed` + `column_value_relation identity` | `columns_unchanged "*" except [old]` + `row_count unchanged` |

Full worked example (filter with nulls kept on purpose):

```yaml
gold_contract:
  contract_version: "1.0"
  must_change:
    - id: c1
      type: rows_removed_where
      target: null
      condition: {column: Age, op: "<", value: 18}
    - {id: c2, type: row_count, target: null, op: decreased}
  must_preserve:
    - id: p1
      type: rows_retained_where
      target: null
      condition:
        any:
          - {column: Age, op: ">=", value: 18}
          - {column: Age, op: is_null}
    - {id: p2, type: columns_unchanged, target: "*", except: []}
```

---

## 12. Known limitations and differences from the guide

### 12.1 Limitations of v1.0

| Limitation | Effect | Tracked in |
|---|---|---|
| No predicate forbids **extra columns** (`columns_unchanged "*"` only covers before columns) | Leftover helper columns pass. | OI-7 |
| New columns are not typed unless a `column_dtype` is added | A new int column computed as float passes. | OI-7 |
| No nested expressions, column-to-column conditions, datetime arithmetic, regex, or group statistics | Some requests need data-dependent literals (medians, modes) or can't be written. | §6 |
| Literals computed from one file (for example median 28.0) | The contract only holds for that snapshot. | task `notes` |
| Index alignment at the same row count tolerates up to 10% disagreeing rows | A small reorder followed by `reset_index` is reported as "values changed", not "reordered" (still a VIOLATION). | §9.2 |
| Time-zone-aware datetimes are their own family (`datetime_tz`) | `dtype: datetime` doesn't accept them. | §7 |

### 12.2 Where the guide (§5) differs from the code

| Guide says | Code does |
|---|---|
| Floats use `atol` and `rtol` (default 0) | Only `atol` exists. |
| `cast` to datetime parses ISO `YYYY-MM-DD` by default | It parses any ISO 8601 text (date, or date and time). |
| `map`: unlisted values unchanged | Also looks a value up by its string form before leaving it unchanged. |
| `categories` order matters only if `ordered: true` | Order also matters if the after column itself is ordered. |
| Canonical condition form is required in gold contracts | Matching canonicalises automatically, so the form is a style choice only. |

### 12.3 Changelog

| Version | Date | Change |
|---|---|---|
| 0.1 | — | Initial draft from the proposal plus extensions. |
| 0.2 | — | Hand-written variants removed; candidates are LLM outputs only. |
| 1.0 | 2026-10-29 | Frozen, including `columns_added`, `columns_removed`, `rows_retained_where`. |

After 1.0, any change needs team agreement, a version bump, and re-checking every task that uses the
changed predicate.
