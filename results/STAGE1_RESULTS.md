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
