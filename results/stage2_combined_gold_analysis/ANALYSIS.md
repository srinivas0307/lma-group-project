# Combined 3B with gold contracts: completed analysis

The completed 100-task gold run achieves **84 correct accepted outputs, zero incorrect acceptances and 16 withheld**. Both matched predicted-contract methods achieved 74 correct, one incorrect accepted and 25 withheld. Replacing the contract source therefore adds ten correct completions, eliminates the observed wrong acceptance and reduces withholding by nine in each comparison. It does not eliminate generation/repair failures.

## Completion, notebook error and validation

The uploaded ZIP has 100 task records, a complete seal, valid hashes and no inference failure record. Generator and repair are the intended pinned Qwen2.5-3B-Instruct revision, with verified unquantized FP16 parameters, on T4 x2. No parser or adapter was loaded. Python, software versions, input hashes, task inventory, model policy, generator prompt, repair prompt, budgets and schema guard match the two completed combined predicted-contract runs.

The notebook source matches its template. Its final summary cell raises `ValueError: Gold contract differs from task` **after all 100 inference tasks completed**. This is a collector implementation defect: `taxi_col_002` contains numeric mapping keys in its YAML gold contract; JSON converts those keys to strings. The original collector compares the JSON-loaded object against the unconverted Python object. All other task gold objects compare directly; this one compares exactly after the expected JSON round trip.

The external compatibility collector at `tools/collect_stage2_combined_gold.py` corrects only that authenticity comparison. It retains every seal, runtime, source, task, settings, contract and cost check. It runs against the exact original source pack; no sealed experiment file or uploaded result has been edited. A regression test checks the numeric-key case and verifies that a genuinely changed contract is still rejected. The full GPU job does not need to be repeated.

CPU replay of all 100 references, all initial programs and all recorded final programs finds **zero generation-input, reference-verdict, initial-correctness or accepted-final mismatches**. Local CPU Python is 3.12, while recorded Kaggle Python is 3.13.16; this agreement covers these observed cases, not arbitrary cross-version behavior. See `audit.json` and `comparison/summary.json`.

## What this establishes about the bottleneck

The gold run has **77 initially correct programs, zero initially correct programs lost, and seven repaired programs: 77 − 0 + 7 = 84**. Final outcomes are 84 ACCEPT, 15 REJECT and one ESCALATE. All 100 correct references pass both original gold verification and the added schema guard. Every withheld task started with an incorrect program; none of the 16 recorded final candidates is correct under the benchmark equivalence checks.

This is direct evidence for two distinct problems:

1. Predicted contracts materially damage the end-to-end result. Selective facts lost five initially correct outputs and tools lost nine; gold loses none. Gold also removes parse stops and the shared wrong-column acceptance. However, the matched difference measures the total contract-source effect on generation and verification/repair, not a count of parser errors that can each be recovered independently.
2. Even supplied gold contracts leave 16 generation/repair failures. The 3B loop repairs seven of its 23 initially incorrect programs (30.4%). These remaining outcomes cannot be blamed on the intent parser, which was bypassed. They demonstrate failure to produce a correct accepted program within the current repair procedure and budget.

There is no evidence in this run that the verifier rejected the correct reference or discarded an initially correct program. That does not prove general verifier completeness, annotation correctness or absence of every intermediate-candidate problem. In particular, rounding follows the benchmark's declared NumPy convention and preservation includes dtype families; these are specification choices to describe in the report.

## Paired gains and regressions

For each predicted method, gold solves 13 tasks the predicted run does not, while the predicted run solves three that gold does not: net +10. The three common regressions are `adult_col_002`, `adult_dedup_003` and `taxi_filt_005`. Different contracts change initial programs and repair trajectories, so gold is not a guaranteed task-by-task upper bound.

The shared wrong acceptance, `airbnb_filt_002`, becomes a correct accepted output with gold. The full gain/loss task IDs are in `comparison/summary.json`.

## The 16 remaining cases

- Column transformations: `adult_col_002` reconstructs the preserved hours-per-week column with float dtype; `titanic_col_001` removes required source columns and later references SibSp; `titanic_col_003` fails explicit two-decimal rounding, eventually calling NumPy round without decimals.
- Type conversions: `adult_conv_003` uses nonexistent education_num and fails to assign category changes; `airbnb_conv_005` calls unsupported categorical arguments rather than assigning the ordered categorical result.
- Deduplication: `adult_dedup_003` retains an undeclared helper column; `airbnb_dedup_002`, `airbnb_dedup_004`, `airbnb_dedup_005`, `taxi_dedup_005` and `titanic_dedup_002` fail to retain exactly the required first/last original records. For Airbnb host/neighbourhood deduplication, an unrequested last_review null filter loses required records from 36 groups.
- Filtering: `taxi_filt_004` changes the original timestamp dtype; `taxi_filt_005` removes required rows and payment_type; `titanic_filt_003` removes 177 unknown-age passengers who should remain; `titanic_filt_005` ends with invalid Python syntax.
- Missing values: `titanic_miss_005` uses broken grouped-median merge/column logic and ends with an Age_x KeyError.

The final candidates contain five execution/syntax failures and eleven runnable but incorrect outputs. These are concrete repair targets, not grounds for relaxing acceptance checks. The helper-column guard prevents an incorrect acceptance in the gold deduplication case.

Correct results by transformation family are column transformations 17/20, deduplication 14/20, filtering 16/20, missing values 19/20 and type conversion 18/20.

## Cost and historical gold comparison

Gold uses 327,153 generation/repair tokens and zero parser tokens. Selective facts uses 379,210 generation/repair plus 177,514 parser tokens; tools uses 426,969 plus 258,796. Gold is an oracle with free supplied contracts, so its lower total cost is not a deployable efficiency result. Summed gold task time is about 12.1 minutes, excluding setup, model loading and reference scoring.

The historical gold configuration achieved 94 correct, zero wrong acceptances and six withheld using 7B NF4 repair and the original loop. Current gold achieves 84/0/16 using combined 3B FP16. Eleven tasks succeed only historically and one succeeds only now. Initial program text matches on only 51/100 tasks, despite both runs having 77 initially correct outputs. Therefore the ten-point historical difference is not an isolated model-size effect: repair model/precision, repair prompting, guard and generation/runtime inputs differ between protocols. Do not attribute the whole difference to the 3B model or to the combined prompt alone.

## Recommended next step

Treat contract grounding and 3B repair as two development tracks. The matched gold result supports pursuing the parser improvements, but parser improvements alone cannot explain away the 16 failures with gold. For repair, test focused source-column preservation, pandas category assignment, explicit rounding arguments, input-order deduplication, grouped transform/fill and syntax/API checks on separate development cases. Keep the current runs frozen and compare new variants against controls. Do not retrain, increase budgets or weaken the verifier merely to improve these observed scores.

Reproduce collection and CPU analysis with `.venv/bin/python tools/analyze_stage2_combined_gold.py`. This uses saved model outputs only. The original uploaded ZIP and executed notebook are preserved under `kaggle_outputs/stage2/combined_gold_v1/` and `notebooks/executed/stage2/combined_gold_v1/`.
