# Stage 1 results

**Read this first.** The main Stage 1 table is the **pool v4** section right below. Everything after it is history
(pool v2: adult + airbnb only; pool v3: four datasets from four partial generation runs); those files are kept
unchanged in `archive/`.

Where the current (pool v4) files are:

| System | Folder |
|---|---|
| Baseline 1 (no verification) | `results/stage1/baseline1/` |
| Baseline 2 (conventional checks, lenient tie-break) | `results/stage1/baseline2/` |
| Baseline 3 (prompted 7B, p1-v3) | `results/stage1/baseline3/` |
| Oracle (gold contracts) | `results/stage1/oracle/` |
| Size ablation (prompted 3B p1-v2, 1.5B p1-v3) | `results/stage1/size_ablation/` |
| Parser runs not selected on dev | `results/stage1/other_prompts/` |
| All systems side by side, prompt selection | `results/stage1/summary/` |

Regenerate pool v4 (CPU, about 20 min):
```
.venv/bin/python eval/build_pool.py --version v4 --sources kaggle_outputs/generation/3b_all_tasks kaggle_outputs/generation/1_5b_all_tasks  # refuses if it exists
.venv/bin/python eval/baseline1_by_dataset.py
.venv/bin/python eval/baseline2.py                  # configs/baseline2_v4.yaml
.venv/bin/python eval/run_intent_oracle.py          # gold contracts -> results/stage1/oracle
.venv/bin/python eval/score_parsers.py              # every kaggle_outputs/parsing/* run + results/stage1/summary/
```

## Pool v4: the main Stage 1 table (2026-10-30)

Pool v4 (`benchmark/pool/`) is built from **one all-task Baseline 1 run per generator**: 3B
(`kaggle_outputs/generation/3b_all_tasks/`, notebook `notebooks/executed/baseline-1_all-tasks_ran-notebook.ipynb`) and
1.5B (`kaggle_outputs/generation/1_5b_all_tasks/`, notebook `notebooks/executed/baseline-1_1.5B_all-tasks_ran-notebook.ipynb`).
Same prompt (`ebbe8bceae71`), seeds, sampling and pinned pandas as pool v3; the prompt schema is identical to the frozen one.
Reason for the switch: one Baseline 1 run per model over all 120 tasks, instead of four partial runs.

**Difference from pool v3** (archived in `archive/pools/pool_v3/`, its results in `archive/results/stage1_pool_v3/`):
1438 of 1440 generations are identical. Two greedy outputs differ because greedy decoding runs in batches of 8 and
the batches were made up differently (padding and fp16 arithmetic): `taxi_conv_005` (3B; pool v3 had a wrong
version, v4 has a correct one equal to an existing sample) and `titanic_conv_001` (1.5B; only the docstring
differs, correct in both). Result: 803 candidates (v3: 804), test runnable wrong 213 (v3: 214). 798 candidates keep
the same id, code and label; the rest are the renumbered `taxi_conv_005` candidates and the `titanic_conv_001` one.
Baseline 2 re-tuned on v4 dev gives the same thresholds (the changes are in test). Parser contracts are per task,
so the 9 parser runs were re-scored without any new GPU run.

| System (test, 671 candidates) | Recall | FRR | Silent error rate | ESCALATE |
|---|---|---|---|---|
| Baseline 1: no verification | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.338 | 0.000 |
| Baseline 2: conventional checks | 0.455 [0.33, 0.58] | 0.139 [0.07, 0.22] | 0.244 | 0.000 |
| **Baseline 3: prompted 7B (p1-v3)** | 0.892 [0.82, 0.95] | 0.254 [0.16, 0.36] | 0.069 | 0.148 |
| prompted 3B (p1-v2, dev-selected) | 0.958 [0.90, 0.99] | 0.578 [0.46, 0.68] | 0.049 | 0.297 |
| prompted 1.5B (p1-v3) | 0.953 [0.90, 0.99] | 0.796 [0.71, 0.88] | 0.105 | 0.356 |
| Intent verifier, oracle contract | 0.944 [0.90, 0.98] | 0.000 [0.00, 0.00] | 0.028 | 0.006 |

Brackets: 95% task-level bootstrap intervals. Paired McNemar (test): Baseline 3 vs Baseline 2: Baseline 3 right on
140 candidates where Baseline 2 was wrong, the reverse on 95 (**p = 0.004**); oracle vs Baseline 2: 164 vs 2
(p < 0.001). All rates moved by at most 0.002 from pool v3, so the findings in the pool v3 history section
(Baseline 2 weak on type conversion, the oracle's 12 misses: 8 extra columns OI-7, 4 taxi dtype annotation gaps) hold.

Baseline 1 by dataset (`eval/baseline1_by_dataset.py` -> `results/stage1/baseline1/by_dataset.json`), test greedy
accuracy: 3B adult 20/25, airbnb 21/25, taxi 18/25, titanic 18/25 (77/100); 1.5B 20, 17, 20, 19 (76/100). These
equal the two notebooks' own `test.greedy` numbers.

---

# History: pool v2, oracle (gold) contracts (MVP-1, README Phase 5)

Pool v2 (`archive/pools/pool_v2/`): 395 unique LLM candidates (Qwen2.5-3B + Qwen2.5-1.5B, prompt `b1-v1`), labelled
correct/wrong by content comparison with the reference output. All systems judge the **same** candidates.

Regenerated at the time with (paths updated to the archive):
```
.venv/bin/python eval/baseline2.py --config archive/configs/baseline2_v2.yaml
.venv/bin/python eval/run_intent_oracle.py --pool archive/pools/pool_v2/pool_v2.jsonl --out archive/results/intent_oracle_v2
.venv/bin/python eval/metrics.py --pool archive/pools/pool_v2/pool_v2.jsonl \
  --verdicts archive/pools/pool_v2/verdicts_baseline1.jsonl archive/results/baseline2_v2/verdicts_baseline2.jsonl \
             archive/results/intent_oracle_v2/verdicts_intent_oracle.jsonl \
             archive/results/intent_oracle_v2/verdicts_intent_oracle_change_only.jsonl \
             archive/results/intent_oracle_v2/verdicts_intent_oracle_preserve_only.jsonl \
  --out archive/results/intent_oracle_v2/metrics_stage1.json
```

## Main table (test split: 323 candidates, 295 ran, 98 of those wrong, 197 correct)

| System | Recall (wrong rejected) | False-rejection rate | Silent error rate (wrong among accepted) | Accepted |
|---|---|---|---|---|
| Baseline 1: no verification | 0.000 | 0.000 | 0.332 [0.22, 0.44] | 295 |
| Baseline 2: conventional checks | 0.510 [0.36, 0.64] | 0.081 [0.02, 0.17] | 0.210 [0.13, 0.30] | 229 |
| **Intent verifier, oracle contract** | **0.980 [0.94, 1.00]** | **0.000** | **0.010 [0.00, 0.03]** | 199 |
| Ablation: must_change only | 0.847 [0.72, 0.94] | 0.000 | 0.071 | 212 |
| Ablation: must_preserve only | 0.347 [0.20, 0.52] | 0.000 | 0.245 | 261 |

Brackets: 95% task-level bootstrap intervals. Paired exact McNemar on "decision was right":
oracle vs Baseline 2: oracle right on 62 candidates where Baseline 2 was wrong, the reverse on 0 (p < 0.001).
ESCALATE (UNCERTAIN): 1 of 323 test candidates (`adult_conv_002__c01`, wrong: the target column was renamed);
1 more in dev. Both count as "not accepted".

## By transformation type (test, recall on runnable wrong candidates)

| Type | Wrong | Baseline 2 | Oracle | Change only | Preserve only |
|---|---|---|---|---|---|
| deduplication | 53 | 0.528 | 1.000 | 1.000 | 0.113 |
| filtering | 20 | 0.700 | 1.000 | 0.600 | 0.850 |
| type_conversion | 14 | 0.357 | 1.000 | 0.857 | 0.571 |
| column_transformation | 7 | 0.429 | 0.857 | 0.714 | 0.143 |
| missing_values | 4 | 0.000 | 0.750 | 0.250 | 0.500 |

Column transformation and missing values have few errors (7, 4): low confidence (OI-1).
Baseline 2's false rejections are concentrated in deduplication (0.320) and filtering (0.157); the oracle has none.

## Both directions are needed (the "bidirectional" claim, README G4)

- must_change alone misses 15% of errors, mostly filtering (0.600): code that removes the right rows **and more**
  satisfies "these rows are gone" but violates "these rows are kept".
- must_preserve alone misses 65%: it can't see that the requested change didn't happen
  (e.g. duplicates left, wrong keep rule, wrong category order).
- Together: 0.980.

## Pool v1 (3B only, for comparison)

Test: Baseline 2 recall 0.469 / FRR 0.099; oracle recall 1.000 / FRR 0.000 / silent error rate 0.000.

## Misses (2 of 98 test errors) — vocabulary gap, not a verifier bug

| Candidate | Why the answer key says wrong | Why the contract passes it |
|---|---|---|
| `adult_miss_005__c07` | adds an extra column `is_not_working` | no predicate forbids extra columns (`columns_unchanged: "*"` only covers before-columns) |
| `airbnb_col_001__c03` | new column is float, reference is int | the contract has no `column_dtype` for the new column |

The gold contracts were **not** changed after seeing these (that would leak test information). See OI-7.

## Caveats (state these in the write-up)

1. **FRR 0 is close to guaranteed by construction.** "Correct" means content-equal to the reference, and every
   reference passes its gold contract (tested, also after `reset_index` and row reversal). So the oracle FRR only
   measures robustness to index/order differences. Correct-but-different interpretations are labelled *wrong* by the
   answer key, so they can't show up as false rejections. A parsed (non-gold) contract will not have this property.
2. **The gold contracts are drafts by one annotator** (`status: draft`, `reviewer: null`). The same intent was used
   to write the reference code and the contract. The second annotation and agreement study (Phase 2) are what make
   the oracle numbers credible; until then treat them as an upper bound.
3. **Baseline 2 sees no request by design** (README risk table): this is the comparison the research question asks for.
4. Candidates are correlated within tasks (36 test tasks contribute wrong candidates); the bootstrap resamples tasks.

## Gate 2 (README §8)

Question: does the oracle-contract verifier beat conventional checks on recall at a similar false-rejection rate?
**Yes:** recall 0.980 vs 0.510 at FRR 0.000 vs 0.081 (test, pool v2), McNemar p < 0.001. Proceed to Phase 6
(Baseline 3, prompted parser), with caveats 1–2 above.

## Baseline 3: prompted parsers, prompt p1-v1 (first run, 2026-10-29)

Same pool (v2, test). Contracts come from the parser instead of the gold annotation. `eval/score_parsers.py`.

| System | Invalid contracts (tasks) | Change F1 | Preserve F1 | Recall | FRR | ESCALATE rate |
|---|---|---|---|---|---|---|
| Baseline 2 (conventional) | – | – | – | 0.510 | 0.081 | 0.000 |
| intent_oracle (gold) | 0 | 1 | 1 | 0.980 | 0.000 | 0.003 |
| intent_prompted_7b (Baseline 3) | 0.340 | 0.720 | 0.771 | 1.000 | 0.477 | 0.337 |
| intent_prompted_3b | 0.440 | 0.403 | 0.660 | 0.949 | 0.594 | 0.331 |
| intent_prompted_1_5b | 0.500 | 0.329 | 0.588 | 0.949 | 0.782 | 0.477 |

Contract metrics: test split, 100 tasks. Recall/FRR: test split, 323 candidates. Invalid or unusable contracts are
escalated (count as "not accepted"), which is where most of the false rejections come from.

**Findings**
1. **Invalid output is the main failure.** 34–50% of the contracts are rejected by the schema, mostly for format:
   pandas dtype names (`int64`) instead of the six families, `and`/`or` instead of `all`/`any`, several arguments
   merged into one object, functions outside the vocabulary (`median`), `ordered` on non-category dtypes, `keep` with
   the wrong op. Larger model → fewer invalid outputs and higher F1 (7B > 3B > 1.5B on every metric).
2. **When the contract is valid it is useful but too strict.** On test tasks with a valid 7B contract
   (66 tasks, 205 candidates): recall 1.000, FRR 0.226, ESCALATE 0.093 (oracle on the same candidates: 0.986 / 0.000;
   Baseline 2: 0.514 / 0.120). Precision is high (change 0.93, preserve 0.99) and recall is low (0.59 / 0.63): the
   predicates the parser writes are mostly right, but it leaves some out and writes some wrong ones that reject correct code.
3. **Baseline 3 does not beat Baseline 2 overall** (McNemar: Baseline 2 right on 89 candidates where Baseline 3 was
   wrong, the reverse on 59; p = 0.017), because of the escalations. The intent approach needs a reliable parser:
   this is the motivation for fine-tuning (Gate 3) and is a result in itself.

**Next:** prompt p1-v2 adds explicit format rules, each one aimed at an invalid-output type that **also occurs on the
dev split** (dev invalid outputs: 7B 7/20, 3B 9/20, 1.5B 15/20). Few-shot examples, decoding and verifier are unchanged.
Both versions are reported. Caveat: the test results of p1-v1 had been seen when p1-v2 was written; the changes are
limited to format rules justified by dev errors, not to content.

## Baseline 3, prompt p1-v2 (format rules from dev errors), 2026-10-30

Same models, few-shot examples, decoding, verifier and pool. Test split.

| System | Invalid (test) | Invalid (dev) | Change F1 | Preserve F1 | Recall | FRR | ESCALATE |
|---|---|---|---|---|---|---|---|
| Baseline 2 | – | – | – | – | 0.510 | 0.081 | 0.000 |
| intent_prompted_7b (p1-v1) | 0.340 | 0.350 | 0.720 | 0.771 | 1.000 | 0.477 | 0.337 |
| **intent_prompted_7b_p1v2** | **0.330** | 0.250 | 0.709 | 0.773 | 0.949 | **0.391** | 0.279 |
| intent_prompted_3b (p1-v1) | 0.440 | 0.450 | 0.403 | 0.660 | 0.949 | 0.594 | 0.331 |
| intent_prompted_3b_p1v2 | 0.290 | 0.350 | 0.411 | 0.781 | 0.969 | 0.487 | 0.285 |
| intent_prompted_1_5b (p1-v1) | 0.500 | 0.750 | 0.329 | 0.588 | 0.949 | 0.782 | 0.477 |
| intent_prompted_1_5b_p1v2 | 0.440 | 0.600 | 0.367 | 0.658 | 0.980 | 0.731 | 0.402 |

Paired McNemar (test): p1-v2 better than p1-v1 for 7B (21 vs 9 candidates, p = 0.043) and 3B (27 vs 4, p < 0.001).
Baseline 2 vs 7B p1-v2: 73 vs 55, p = 0.13 (no longer significantly worse, not better).

**Findings:** format rules cut invalid outputs (most for 3B: 44% → 29%) and false rejections, but 29–44% of contracts
are still invalid. The remaining errors are the same kinds (condition ops outside the vocabulary, merged arguments,
pandas dtype names, `keep` with the wrong op): small prompted models don't follow a strict schema reliably.
**Update 2026-10-30:** a coverage gap was found by inspecting the prompt: the few-shot examples demonstrate only 1 of 14
functions and 4 of 10 condition operators. Prompt p1-v3 adds a reference card covering the whole vocabulary (design
rule, README §12). The final Baseline 3 version is chosen per model on the **dev** split only
(`archive/results/parsers_v2/baseline3_selection.json`); all three versions are reported. Caveat: p1-v1 and p1-v2 test results
had been seen, and the gap was quantified on test (7B p1-v2: 57% invalid when the gold contract needs a function or
operator the examples don't show, vs 14% otherwise). Remaining format reliability is addressed the same way for every parser (OI-9) and is the main motivation
for fine-tuning.

## Baseline 3, final (prompt p1-v3 + dev-only selection), 2026-10-30

**Verifier fix first (applies to every system; found while analysing these results):** names in `except` of
`columns_unchanged: "*"` were checked as column references, so a contract that excluded a column the transformation
**adds** (e.g. `except: ["net_capital"]`) became UNCERTAIN (`unknown_column`). Per guide §5.5, `except` only removes
names from the expansion, so a name that isn't in the input removes nothing. Fixed in `predicate_columns`; oracle and
Baseline 2 verdicts are byte-identical after the fix (gold contracts never do this); every prompted system was rescored.

All prompted parsers, test split (pool v2; contract metrics on the 100 test tasks):

| System | Invalid | Change F1 | Preserve F1 | Recall | FRR | ESCALATE |
|---|---|---|---|---|---|---|
| intent_prompted_7b | 0.340 | 0.720 | 0.771 | 0.980 | 0.391 | 0.279 |
| intent_prompted_7b_p1v2 | 0.330 | 0.709 | 0.773 | 0.939 | 0.305 | 0.223 |
| intent_prompted_7b_p1v3 | 0.130 | 0.751 | 0.906 | 0.969 | 0.228 | 0.192 |
| intent_prompted_3b | 0.440 | 0.403 | 0.660 | 0.949 | 0.569 | 0.316 |
| intent_prompted_3b_p1v2 | 0.290 | 0.411 | 0.781 | 0.969 | 0.442 | 0.257 |
| intent_prompted_3b_p1v3 | 0.220 | 0.525 | 0.831 | 0.969 | 0.360 | 0.204 |
| intent_prompted_1_5b | 0.500 | 0.329 | 0.588 | 0.949 | 0.782 | 0.477 |
| intent_prompted_1_5b_p1v2 | 0.440 | 0.367 | 0.658 | 0.980 | 0.731 | 0.402 |
| intent_prompted_1_5b_p1v3 | 0.340 | 0.395 | 0.730 | 0.990 | 0.741 | 0.319 |

**Selection on dev** (`archive/results/parsers_v2/baseline3_selection.json`, lowest dev invalid rate, then dev F1):
7B → **p1-v3**, 3B → **p1-v2**, 1.5B → **p1-v3**. The 3B choice follows the rule even though p1-v3 is better on test
(dev invalid 0.35 vs 0.40); report both.

**Main table (test):**

| System | Recall | FRR | Silent error rate | ESCALATE |
|---|---|---|---|---|
| Baseline 1 | 0.000 | 0.000 | 0.332 | 0.000 |
| Baseline 2 | 0.510 | 0.081 | 0.210 | 0.000 |
| **Baseline 3 = intent_prompted_7b_p1v3** | **0.969** | **0.228** | 0.019 | 0.192 |
| intent_oracle (upper bound) | 0.980 | 0.000 | 0.010 | 0.003 |

Paired McNemar (test): Baseline 3 right on 61 candidates where Baseline 2 was wrong, the reverse on
45 (p = 0.1448): **not significant**. Oracle vs Baseline 3: oracle better on 66 vs 1.

**Findings**
1. The reference card (p1-v3) halved the 7B's invalid outputs on test (33% → 13%) and raised preserve F1 0.77 → 0.91,
   as predicted by the coverage analysis.
2. Prompted intent verification catches almost every wrong candidate (recall 0.97 vs 0.51 for conventional checks),
   but rejects ~23% of correct ones, mostly because of escalated invalid contracts and predicates the parser writes
   wrongly. Net, it is not significantly better than Baseline 2. The gap to the oracle (FRR 0.23 → 0.00) is the room
   the fine-tuned parser has to close (Gate 3).
3. Bigger is better: at its selected prompt, 7B > 3B > 1.5B on invalid rate, F1 and FRR.

## History: pool v3, all four datasets (superseded by pool v4; files in `archive/pools/pool_v3/`, `archive/results/stage1_pool_v3/`)

Pool v3 (`benchmark/pool/`) = pool v2 (adult, airbnb; all 395 candidates kept with the same ids and labels) + taxi and
titanic generated with the same two models, prompt (`ebbe8bceae71`), sampling, seeds rule and library versions
(transformers 5.16.1, torch 2.11, T4). 804 unique candidates; test: 672 candidates, 631 ran, **214 runnable wrong**
(adult 48, airbnb 50, taxi 66, titanic 50) over 120 tasks. Baseline 2 re-tuned on v3 dev with the same config
(`configs/baseline2_v3.yaml`). Parser contracts are the same runs as before (they covered all 120 tasks).

| System (test, 672 candidates) | Recall | FRR | Silent error rate | ESCALATE |
|---|---|---|---|---|
| Baseline 1: no verification | 0.000 [0.00, 0.00] | 0.000 [0.00, 0.00] | 0.339 | 0.000 |
| Baseline 2: conventional checks | 0.453 [0.33, 0.57] | 0.139 [0.07, 0.22] | 0.246 | 0.000 |
| **Baseline 3: prompted 7B (p1-v3)** | 0.893 [0.82, 0.95] | 0.254 [0.16, 0.36] | 0.069 | 0.147 |
| prompted 3B (p1-v2, dev-selected) | 0.958 [0.90, 0.99] | 0.578 [0.46, 0.68] | 0.049 | 0.298 |
| prompted 1.5B (p1-v3) | 0.953 [0.90, 0.99] | 0.796 [0.71, 0.88] | 0.105 | 0.357 |
| Intent verifier, oracle contract | 0.944 [0.90, 0.98] | 0.000 [0.00, 0.00] | 0.028 | 0.006 |

Paired McNemar (test): Baseline 3 vs Baseline 2: Baseline 3 right on 141, Baseline 2 on
95 (**p = 0.0033**); oracle vs Baseline 2: 165 vs 2 (p < 0.001);
Baseline 2 vs Baseline 1: 97 vs 58 (p = 0.0022).

**Findings (v3 vs v2)**
1. With twice the tasks, **Baseline 3 is now significantly better than Baseline 2** (p = 0.003): it catches 0.89 of the
   wrong candidates vs 0.45, at FRR 0.25 vs 0.14. On pool v2 alone the difference was not significant (p = 0.14).
2. Baseline 2 gets weaker on the new data (recall 0.51 → 0.45, FRR 0.08 → 0.14); it catches **no** type-conversion
   errors (0/30) and 15% of missing-value errors.
3. The oracle stays at FRR 0.000 with recall 0.944. Its 12 misses are two gaps, both known before this run:
   - 8 × an extra helper column left in the output (e.g. taxi code that adds `pickup_datetime` to dedup on it):
     the vocabulary has no "no extra columns" predicate (OI-7). Impact grew from 2/98 to 8/214.
   - 4 × `passenger_count` turned into `object` dtype by `replace(0, pd.NA)`: the gold contracts of taxi_miss_001/005
     don't state the column's dtype (annotation gap, OI-8). Gold contracts are **not** changed after seeing test.

### Baseline 1 (direct LLM, no verification) by dataset, pool v3

Baseline 1 has no verifier, so it needs no separate run: its result is the generator output itself. These numbers
come from the generations already made on Kaggle for pool v3 (both generators, all four datasets, prompt
`ebbe8bceae71`), each labelled with its pool candidate's correctness. Regenerate:
`.venv/bin/python eval/baseline1_by_dataset.py` (-> `results/stage1/baseline1/by_dataset.json`).

Test split, 25 tasks per dataset. Greedy = one temperature-0 output per task; samples = all 6 outputs per task
(greedy + 5 at T = 0.7). Silent error rate = wrong among the unique pool candidates that ran (the Baseline 1 row above).

| Test | adult | airbnb | taxi | titanic | all |
|---|---|---|---|---|---|
| Qwen2.5-3B greedy accuracy | 20/25 (0.80) | 21/25 (0.84) | 17/25 (0.68) | 18/25 (0.72) | 76/100 (0.76) |
| Qwen2.5-3B sample accuracy | 0.800 | 0.827 | 0.740 | 0.720 | 463/600 (0.772) |
| Qwen2.5-1.5B greedy accuracy | 20/25 (0.80) | 17/25 (0.68) | 20/25 (0.80) | 19/25 (0.76) | 76/100 (0.76) |
| Qwen2.5-1.5B sample accuracy | 0.733 | 0.693 | 0.720 | 0.693 | 426/600 (0.710) |
| Silent error rate (unique candidates) | 48/154 (0.312) | 50/141 (0.355) | 66/176 (0.375) | 50/160 (0.312) | 214/631 (0.339) |

**Baseline 1 notebook rerun on all four datasets (2026-10-30).** The Baseline 1 notebook was rerun in one go for 3B
on all 120 tasks (`notebooks/executed/baseline-1_all-tasks_ran-notebook.ipynb`, outputs in
`kaggle_outputs/generation/3b_all_tasks_rerun/`; the first adult + airbnb-only run is in `archive/notebooks/`). Same
prompt (`ebbe8bceae71`), seeds, sampling and pinned pandas; the prompt schema is identical to the frozen one.
**719 of 720 generations are identical** (same text, same label) to the 3B generations in pool v3. The one difference
is `taxi_conv_005` greedy: the rerun's code adds `.astype({'store_and_fwd_flag': bool})` and is correct; the pool's
version is wrong (object dtype). Cause: greedy outputs are generated in batches of 8, and the rerun batches all 120
tasks while the pool run batched taxi + titanic only, so the padding and fp16 arithmetic differ for that batch.
Its test greedy accuracy is therefore 77/100 (taxi 18/25) against 76/100 (taxi 17/25) in the table above.
Pool v3 is **not** changed: Baselines 2, 3 and the oracle judged the pool's candidates, and swapping one test
candidate after seeing the result would be selection on test. Report the table above (from the pool) as Baseline 1,
and the rerun as a reproducibility check (1/720 generations differ, from batched fp16 greedy decoding). Even greedy decoding of the stronger generator gets about 1 task in 4
wrong, and roughly a third of the distinct programs that run are silently wrong.
