# Stage 2 failure analysis

## Conclusion

The current proposed system trades more abstentions for fewer incorrect acceptances. It does **not** demonstrate higher end-to-end task accuracy than direct generation: 76/100 versus 78/100. This is a valid mixed research result, but it does not support a claim that the proposed system is more accurate overall.

The main observed bottleneck is the fidelity of predicted contracts. Incorrect contracts reject correct programs and can accept programs implementing the wrong transformation. Conventional checks have a different problem: a generic distribution-shift threshold rejects legitimate requested filtering and deduplication. There are also genuine generation/repair failures.

The evidence below comes from the completed, frozen `stage2-clean-v1` experiment. No models were called or prompts, contracts, checker thresholds, or recorded results changed during this analysis.

## Evidence and reproducibility

- [Original metrics and paired comparisons](stage2_clean_v1/stage2_metrics_test.json): six systems, the same 100 tasks, seed 0; 600 records are not 600 independent tasks.
- [Detailed offline audit](stage2_failure_analysis/audit.json): per-system transitions, per-task reference/checker probes, saved candidate code, correctness comparisons, and verifier failures.
- [Audit script](../tools/analyze_stage2_failures.py): `.venv/bin/python tools/analyze_stage2_failures.py`.
- All six stored runs pass the strict bundle validator. All 218 frozen source/input hashes verify.
- The audit runs all 100 reference programs in the existing sandbox, applies conventional, prompted-contract, fine-tuned-contract and gold-contract checks to their outputs, and replays submitted candidates for all 24 proposed failures, all initially correct outputs lost by the other verification systems, and their incorrect acceptances. The non-proposed replay is targeted, not exhaustive.
- CPU replay uses local Python 3.12 and pandas 2.3.3; Kaggle used Python 3.13.16. Replayed initial labels and accepted-output labels agree with the recorded labels for every audited case. This checks those observed cases, not universal cross-version equivalence.

## What the headline scores mean

Here, **final accuracy** means a correct output was ACCEPTED. A correct candidate that was rejected or escalated scores zero. **Silent error rate** means incorrect accepted outputs divided by all accepted outputs. These metrics answer different questions.

- Direct baseline: **78 correct, 20 incorrect accepted, 2 withheld**. Accepted-output correctness is 78/98 = **79.6%**.
- Conventional checks + repair: **72 correct, 13 incorrect accepted, 15 withheld**. Accepted-output correctness is 72/85 = **84.7%**.
- Prompted 7B contracts + repair: **67 correct, 3 incorrect accepted, 30 withheld**. Accepted-output correctness is 67/70 = **95.7%**.
- Gold contracts: **94 correct, 0 incorrect accepted, 6 withheld**. Accepted-output correctness is 94/94 = **100% on this sample**.
- Proposed system: **76 correct, 2 incorrect accepted, 22 withheld**. Accepted-output correctness is 76/78 = **97.4%**.
- No-contract-to-generator ablation: **73 correct, 2 incorrect accepted, 25 withheld**. Accepted-output correctness is 73/75 = **97.3%**.

Relative to direct generation, the proposed system has 18 fewer incorrect acceptances, 20 additional withheld tasks, and two fewer correct delivered outputs. This is an observed reliability/coverage trade-off, not an unconditional improvement. On individual tasks it gains 13 correct outcomes and loses 15; the net difference hides substantial disagreement.

If the application requires a correct answer for every request, the proposed system has not beaten direct generation. If accepting an incorrect transformation is costly and human review is available, the lower incorrect-acceptance rate may be useful; that benefit must be weighed against the review load and inference cost.

## Where accuracy is lost

The decomposition uses each system's own initial generated code, before verification and repair. Parse-blocked tasks have no initial code.

- Conventional: **78 initially correct − 8 withheld + 2 repaired = 72**.
- Prompted contracts: **69 initially correct − 8 withheld + 6 repaired = 67**, with 13 parse escalations. Nine of those 13 tasks are solved by direct generation. Equivalently, relative to the direct baseline: **78 − 9 parse-blocked successes − 8 verification losses + 6 repairs = 67**.
- Proposed: **72 initially correct − 6 withheld + 10 repaired = 76**, with 2 parse escalations. Both parse-blocked tasks are solved by direct generation.
- No-contract-to-generator: **76 initially correct − 10 withheld + 7 repaired = 73**, with the same 2 parse escalations.
- Gold contracts: **77 initially correct − 0 withheld + 17 repaired = 94**.

The conventional system's initial code is byte-identical to direct generation on all 100 tasks. The prompted-contract and no-contract-to-generator systems also match direct generation on every task where they generate code (87/87 and 98/98). Their losses are therefore attributable to the added parsing/checking/repair paths in these runs, rather than different initial programs.

The proposed system changes the generator prompt and includes predicted contracts, so its initial outputs differ. Do not interpret its 72 initially correct outputs as evidence that the repair agent reduced accuracy: repair and checking together raise its delivered-correct count to 76.

The recorded proposed repair gain of 10 percentage points compares 76 final correct outputs against 66 initially correct outputs accepted by its checker. Its net improvement over all 72 initially correct candidates is four points. Both numbers are useful, but their denominators and meanings must be stated.

## Direct evidence of incorrect checking

When the correct reference output is submitted to each checker:

- Conventional checks: **84 PASS, 16 VIOLATION**.
- Prompted 7B contracts: **73 PASS, 14 VIOLATION, 13 parse failures**.
- Fine-tuned contracts used by proposed and the ablation: **81 PASS, 16 VIOLATION, 1 UNCERTAIN, 2 parse failures**.
- Gold contracts: **100 PASS**.

All 19 fine-tuned-contract tasks that fail this reference check or parsing also fail to deliver a correct accepted output in the proposed run. The other five proposed failures pass the reference check but never produce a correct submitted candidate in the audited trace.

This is a diagnostic association, not a proof that replacing the contracts would recover exactly 19 tasks: changing the contract also changes generation and repair behavior. Likewise, passing the reference output does not prove a predicted contract is complete, and rejecting one reference implementation is not a mathematical proof that every equivalent implementation would be rejected.

### The six initially correct proposed outputs that were withheld

- `adult_conv_002`: the mapping contract uses `<50K` instead of the actual category `<=50K`.
- `airbnb_dedup_001`: the contract deduplicates across all columns instead of the requested listing-name key.
- `taxi_conv_002`: the contract requests datetime conversion while also requiring the original string values to remain unchanged.
- `taxi_miss_004`: the missing `where RatecodeID == 99` condition makes the contract require every rate code to become 1.
- `taxi_miss_005`: one preservation condition is incorrectly shared across both columns, rejecting the requested changes to RatecodeID.
- `titanic_miss_002`: the contract predicts mode `C`; the correct mode is `S`.

The replay confirms that each initial program is correct and its predicted-contract checker rejects it. These are false rejections, not proof that the initial code needs repair.

### Four correct repairs that were rejected

For `airbnb_dedup_003`, `airbnb_filt_005`, `titanic_miss_004`, and `titanic_miss_005`, the initial program is wrong, but a submitted repair is equal to the reference output. The predicted contract still rejects it.

For example, `titanic_miss_004` correctly fills Age with the data's median, **28**, after an initial program incorrectly filled other columns too. The contract demands **37**. The agent repeatedly submits the correct median-fill program, which remains a violation under that contract.

Thus the proposed system encounters correct submitted code in ten withheld tasks: six initial programs and four later repairs. Adding these ten to the final 76 would describe an **offline best-candidate count of 86**, not an achievable accuracy claim or a proposed ground-truth-powered selection rule.

### Both proposed incorrect acceptances follow incorrect contracts

- **`airbnb_filt_002`** asks for Brooklyn and Manhattan listings. The parser chooses `neighbourhood` instead of `neighbourhood_group`. A repair filters the wrong column, returns **0 rows instead of 927**, and passes that wrong contract.
- **`titanic_col_005`** asks to raise ages below 1 to 1. The parser translates this to adding 1 to Age. The generated program increases every nonmissing age by 1 and is accepted, although the reference comparison reports different values.

This is stronger evidence than a vague claim of agent weakness: the verifier is checking the model's incorrect interpretation, and passing those checks is not equivalent to satisfying the user's request.

### Five remaining proposed failures with reference-compatible contracts

`adult_dedup_001`, `adult_dedup_002`, `airbnb_dedup_005`, `taxi_filt_004`, and `titanic_filt_003` have reference outputs that pass the predicted checker. None of their submitted candidates is correct in the replay. These are residual generation/repair failures: wrong deduplication results, unwanted dtype changes, and incorrect filtering. A contract that accepts the reference is not necessarily perfect, so this grouping is evidence of failed search under compatible checks rather than proof that the contract had no influence.

The complete 24-failure grouping is: 2 parse failures; 6 correct initial outputs withheld; 4 correct repairs withheld; 2 incorrect acceptances; 4 other failures with reference-rejecting contracts; 1 reference-uncertain contract; and 5 failures despite reference-compatible contracts.

## Why the conventional baseline loses correct answers

All eight initially correct programs it loses trigger `distribution.mean_shift` with the tuned threshold **0.1**. Filtering and deduplication legitimately change distributions; the check has no representation of which change the user requested.

Example: correctly filtering Airbnb listings changes the mean of `availability_365` by a measured standardized amount of 0.311568, exceeding 0.1. Correct Titanic filtering similarly shifts Age or Pclass. The repair loop is asked to correct an expected consequence of the request.

This explains the conventional system's **78 − 8 + 2 = 72** result. It is a method/threshold mismatch on these tasks, not evidence that the generator suddenly became weaker.

## Statistics, costs, and limits

- Proposed versus direct final correctness: 13 proposed-only successes versus 15 direct-only successes; exact paired McNemar **p = 0.8506**. Neither superiority nor equivalence is established.
- Proposed versus conventional final correctness: 18 versus 14 discordant successes; **p = 0.5966**.
- Proposed versus no-contract-to-generator: 6 versus 3 discordant successes; **p = 0.5078**. The observed three-point gain does not establish a generation-conditioning benefit.
- An exploratory paired analysis of the event "incorrect output accepted" gives 19 direct-only events and 1 proposed-only event, **p ≈ 0.0000401**. This uses all 100 task pairs; it is not a significance test comparing two acceptance-conditioned percentages. It is post hoc and assumes independent task pairs; shared datasets/templates limit generalization.
- The proposed system's mean recorded generation/repair token count is **5,042**, versus **802** for direct generation, about **6.3×**. These counts exclude the earlier saved parser inference, so they are not total end-to-end cost estimates.
- Valid tool arguments do not establish useful tool selection or effective reasoning. Repeated correct code can be rational under a wrong contract, but the loop still consumes calls and often ends in escalation.
- Zero unsafe attempts in these ordinary test tasks is not a replacement for the separate red-team safety evaluation.
- Gold-contract performance is an oracle comparison, not a deployable result. It changes both generator conditioning and verification. The 94% versus 76% gap supports investigating contract quality but does not isolate the causal contribution of the parser alone. Existing gold-annotation review limitations in [OI-8](../docs/open_issues.md#oi-8-oracle-results-depend-on-unreviewed-gold-contracts) still apply.

## What to do next

1. Preserve this completed experiment and report the mixed result directly. A defensible statement is: "On this 100-task benchmark, predicted intent contracts substantially reduced incorrect acceptances, but did not improve accepted-correct task completion over direct generation; incorrect contracts caused false rejections and occasional wrong acceptances."
2. Review contract semantics before investing in a larger benchmark. Priorities suggested by this analysis are data-grounded statistics, correct column/key selection, preservation conditions scoped to the right rows, and contradictions between change and preservation requirements.
3. Develop any changes as a new version on separate development data, including analogous examples rather than patching these test outputs. The current test set is now analyzed and must not be presented as fresh confirmation for fixes derived from these failures.
4. Evaluate targeted existing ideas rather than all extensions: read-only statistic tools for median/mode values, schema/condition consistency checks, and controlled re-asking for invalid contracts. These are hypotheses for development experiments, not demonstrated solutions or guarantees.
5. After freezing the revised method, use an independently reviewed benchmark with diverse datasets and templates. Keep all systems on the same tasks and report accuracy, incorrect acceptances, abstention/review burden, and full inference cost together.

There is no evidence from this audit of a collector or scoring integration error explaining the lower scores. There is clear evidence of semantic contract failures, generic-check false rejections, and unsuccessful code generation/repair. A larger test set alone would measure these issues more precisely; it would not fix them.
