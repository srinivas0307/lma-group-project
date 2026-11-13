# Stage 1: wrong candidates caught, by error kind (test)

213 runnable wrong test candidates of pool v4. Kinds from `benchmark/annotation/error_kinds_pool_v4.csv` (**0 of 213 labels reviewed by a person; the others are drafts**, see `docs/error_taxonomy.md`). Each cell: candidates of that kind the system did not accept / candidates of that kind.

| Kind | n | Baseline 2 (conventional) | Baseline 3 (prompted 7B) | prompted 3B | prompted 1.5B | fine-tuned 1.5B | fine-tuned 3B | Oracle (gold contract) | Oracle, must_change only | Oracle, must_preserve only |
|---|---|---|---|---|---|---|---|---|---|---|
| wrong_target_values | 27 | 11 | 26 | 27 | 26 | 27 | 24 | 27 | 25 | 7 |
| wrong_target_type | 23 | 0 | 18 | 22 | 20 | 19 | 18 | 18 | 9 | 16 |
| removes_too_many_rows | 22 | 20 | 22 | 22 | 22 | 22 | 22 | 22 | 8 | 15 |
| unrequested_change | 20 | 10 | 17 | 20 | 17 | 20 | 20 | 20 | 6 | 20 |
| order_changed_before_keep | 17 | 7 | 17 | 17 | 16 | 17 | 17 | 17 | 17 | 1 |
| helper_column_left | 16 | 3 | 9 | 15 | 14 | 9 | 12 | 9 | 9 | 2 |
| wrong_category_order | 14 | 0 | 9 | 9 | 14 | 14 | 9 | 14 | 14 | 5 |
| removes_too_few_rows | 11 | 1 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 3 |
| wrong_column | 11 | 8 | 10 | 10 | 11 | 10 | 11 | 11 | 8 | 9 |
| inverted_selection | 10 | 9 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 4 |
| groupby_first_mixes_rows | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 0 |
| wrong_output_column | 6 | 3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 4 |
| broken_logic | 6 | 4 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 1 |
| incomplete_key | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1 |
| removes_all_copies | 4 | 3 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 0 |
| wrong_key | 3 | 0 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 0 |
| wrong_proxy_condition | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 3 |
| wrong_null_handling | 2 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 0 | 2 |
| wrong_keep_rule | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 0 |
| hardcoded_value_list | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| incomplete_condition | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| wrong_boolean_logic | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| **all (recall)** | 213 | **0.455** | **0.892** | **0.958** | **0.953** | **0.944** | **0.915** | **0.944** | **0.732** | **0.446** |
