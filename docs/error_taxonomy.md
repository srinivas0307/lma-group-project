# Error taxonomy for wrong LLM candidates (draft v0)

Used to label every runnable wrong candidate in the pool (README Phase 5, task 3). Each candidate gets one
**primary** kind (the mistake that best explains why the output is wrong) and optionally one **secondary** kind.
Labels: `benchmark/annotation/error_kinds_pool_v1.csv` (draft by Kiro, to be reviewed by a person).

| Kind | Meaning | Example from pool v1 |
|---|---|---|
| `wrong_column` | Uses a different column than the request means | dedup by `host_name` instead of `host_id`; filter on `occupation` instead of `workclass` |
| `incomplete_key` | Duplicate key is missing part of what the request names | "one per host and room type" deduplicated by host only |
| `wrong_key` | Duplicate key built wrongly in another way | `df.columns[:-1]` drops `income` from the key instead of `fnlwgt` |
| `order_changed_before_keep` | Sorts before `keep="first"/"last"`, so a different row is kept | `sort_values('fnlwgt').drop_duplicates(..., keep='first')` |
| `removes_all_copies` | Removes every copy of a duplicated row instead of keeping one | `keep=False`, or keeping groups of size 1 |
| `inverted_selection` | Keeps the rows that should be removed (or the reverse) | `df.loc[df.duplicated(...)]` keeps only the duplicates |
| `wrong_boolean_logic` | Negation / and-or error in a compound condition | "remove private rooms in Manhattan under $60" written as "keep Manhattan and not private and >= 60" |
| `incomplete_condition` | Leaves out part of the condition | only `Self-emp-not-inc`, missing `Self-emp-inc` |
| `hardcoded_value_list` | Writes out a list of allowed values from memory instead of using the data | `isin([...13 education levels...])` that misses real levels |
| `wrong_proxy_condition` | Replaces the requested condition with a related column/threshold | `education-num > 4` instead of excluding two named levels |
| `wrong_target_type` | Converts to a different type or encoding than asked | "make income categorical" → 0/1 integers |
| `wrong_category_order` | Ordered category with the wrong order | categories in order of appearance instead of education-num order |
| `wrong_output_column` | Puts the result in a new column and/or drops the original | writes `income_num` and drops `income` |
| `unrequested_change` | Also changes something the request didn't mention | recodes `income` during dedup; converts 3 extra columns to float; renames `id` |
| `helper_column_left` | Leaves a temporary helper column in the output | `first_occurrence`, `last_row`, `keep` |
| `broken_logic` | Code doesn't implement any coherent version of the request | `groupby('host_name')['host_name'].transform(max).eq(df['host_name'])` |

Why it matters: the conventional checks (Baseline 2) only see the data, so they should miss kinds like
`order_changed_before_keep`, `wrong_category_order` and `wrong_target_type`, while contracts that state the
key, the keep rule and the target dtype should catch them. The CSV records Baseline 2's verdict per candidate.
