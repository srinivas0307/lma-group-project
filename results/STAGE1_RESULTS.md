# Stage 1 results: verification only, oracle (gold) contracts (MVP-1, README Phase 5)

Pool v2 (`eval/pool_v2/`): 395 unique LLM candidates (Qwen2.5-3B + Qwen2.5-1.5B, prompt `b1-v1`), labelled
correct/wrong by content comparison with the reference output. All systems judge the **same** candidates.

Regenerate:
```
.venv/bin/python eval/baseline2.py --config configs/baseline2_v2.yaml
.venv/bin/python eval/run_intent_oracle.py
.venv/bin/python eval/metrics.py --pool eval/pool_v2/pool_v2.jsonl \
  --verdicts eval/pool_v2/verdicts_baseline1.jsonl results/baseline2_v2/verdicts_baseline2.jsonl \
             results/intent_oracle_v2/verdicts_intent_oracle.jsonl \
             results/intent_oracle_v2/verdicts_intent_oracle_change_only.jsonl \
             results/intent_oracle_v2/verdicts_intent_oracle_preserve_only.jsonl \
  --out results/intent_oracle_v2/metrics_stage1.json
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
(`results/parsers_v2/baseline3_selection.json`); all three versions are reported. Caveat: p1-v1 and p1-v2 test results
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

**Selection on dev** (`results/parsers_v2/baseline3_selection.json`, lowest dev invalid rate, then dev F1):
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
