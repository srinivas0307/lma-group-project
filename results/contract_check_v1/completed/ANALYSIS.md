# Contract checks and one parser correction: completed GPU runs

Selective facts achieves **76 correct, one incorrect accepted and 23 withheld**, versus its combined control's 74/1/25. Tools achieves **74 correct, one incorrect accepted and 25 withheld**, unchanged from its control. Neither method loses a previously correct final result. Total token use falls 12.4% for selective facts and 8.2% for tools.

These runs provide a modest improvement for selective facts and a token reduction for both methods. They do not establish that the remaining contract problems or wrong acceptance have been solved.

## What changed and whether the comparison is matched

This implementation adds input-only diagnostics after the original parse, followed by at most one parser correction. The selected 1.5B NF4 adapters, 3B FP16 generator, 3B FP16 combined repair, schema guard, generation/repair prompts and budgets are unchanged. It performs no retraining and makes no routing changes. The tool parser's existing query behavior is unchanged; the additional correction asks for a final contract, not a fresh tool conversation.

Both bundles cover all 100 test tasks, have valid CRCs and complete SHA-256 seals, and match source manifest `971a9a25bf8a0860005b409705976159ac58505839ba43c74f043966099ee324`. Both executed notebooks match their templates and have no recorded exception outputs. The source inventory, input hashes, adapters, model revisions, software/Python versions, prompts, budgets and task inventories match the intended configuration. The first extracted contract/status reproduces the corresponding control on **100/100 tasks for each method**.

The full CPU replay, correction-policy validation and observed-label comparison find **zero mismatches across all 200 method/task records** and are recorded in `audit.json`, produced by `tools/audit_contract_check_runs.py`. The paired results are also in `../selective_paired.json` and `../tools_paired.json`. Local CPU Python differs from Kaggle Python; any audit discrepancies are retained explicitly. These previously inspected tasks informed the new rules, so this is diagnostic development evidence, not independent evaluation.

## What actually improved

Selective facts gains exactly two tasks, with no losses:

- `airbnb_dedup_003`: the original contract deduplicates on the wrong key. The diagnostic exposes that those keys have no duplicates; the correction selects a reference-compatible contract and the final output is correct.
- `titanic_conv_003`: the original mapping uses strings True/False as source keys for numeric Survived values 0/1. The correction fixes the mapping, and the final output is correct.

It attempts 14 corrections. Eleven return the same extracted contract/status; three differ. Only two changed replies are selected, and these are the two successful cases. The other changed reply, `taxi_col_005`, remains unacceptable and is withheld. Final reference-compatible contract coverage rises from 83/100 to 85/100.

Tools also attempts 14 corrections: nine replies are unchanged and five differ. Only one changed reply is selected: `airbnb_dedup_001` becomes reference-compatible, but generation/repair still fails, so no final accuracy is gained. Final reference-compatible coverage rises from 84/100 to 85/100. No task gains or loses a correct final result.

The decision counters say `use_corrected` six times for selective facts and five for tools. This does **not** mean six/five useful corrections: the rule accepts a reply with the same number of evidence issues, and several selected replies are exact repeats.

## The remaining incorrect acceptance

Both still incorrectly accept `airbnb_filt_002`, keeping zero rows instead of the correct 927. The diagnostic explicitly reports that Brooklyn and Manhattan occur in neighbourhood_group, not neighbourhood. Both parsers repeat the same wrong neighbourhood contract in the correction turn.

The selection rule allows it because it has no proof-level issue and no more evidence issues than the original. The schema guard cannot detect this semantic column error. Thus detecting the problem is working, but the parser correction and admission policy do not resolve it. This cannot be reported as a fixed wrong-column failure or improved incorrect-acceptance count.

An absent literal is not inherently wrong: legitimate empty filters must remain supported. A future development policy should distinguish repeated unresolved column-grounding evidence from a successfully corrected contract and test explicit clarification/escalation behavior without automatically replacing columns or rejecting every empty result.

## Withholding and the initially-correct metric

Both versions now withhold eight contracts at the parse stage under the diagnostic policy. Selective facts has ten total parse stops, including its two existing no-reply routing failures; tools has eight. Final emitted reference-incompatible contracts decrease to five/seven partly because bad contracts are stopped earlier, not because all were repaired. The remaining withheld totals break down as 10 parse stops + 4 emitted reference-incompatible + 9 reference-compatible failures for selective facts, and 8 + 6 + 11 for tools. The number of reference-compatible contracts, measured over all 100 tasks, is the clearer coverage metric above.

Selective facts: **74 initially correct − 1 lost + 3 repaired = 76**. Tools: **70 initially correct − 2 lost + 6 repaired = 74**.

The controls had 76/77 initially correct and five/nine lost. Do not interpret the smaller lost counts as recovery of all those correct programs: some formerly generated correct candidates are now never generated. Two selective-facts control tasks with initially correct code, airbnb_dedup_001/002, now stop during parsing. Six such tools tasks now stop during parsing: adult_miss_005, airbnb_dedup_002, taxi_conv_002, taxi_miss_003, taxi_miss_004 and titanic_filt_005. Those controls already failed to deliver correct final outputs, so no final success is lost here, but these cases remain unresolved.

## Costs

- Selective facts: parser tokens rise 177,514 → 208,324 (+30,810, 17.4%); generation/repair tokens fall 379,210 → 279,144. Total falls 556,724 → **487,468**, saving 69,256 (12.4%).
- Tools: parser tokens rise 258,796 → 298,416 (+39,620, 15.3%); generation/repair tokens fall 426,969 → 331,111. Total falls 685,765 → **629,527**, saving 56,238 (8.2%).

The extra parser calls are offset by avoiding downstream work on contracts withheld early. This is a useful cost reduction on these runs, but it does not mean all saved work has become successful computation. Summed recorded task time increases from about 33.2 to 33.9 minutes for selective facts and 36.2 to 37.1 minutes for tools; these exclude loading/setup and are not controlled wall-clock comparisons. Diagnostic overhead is not separately itemized for every decision branch, limiting complete latency attribution.

The tool parser still makes eight tool attempts, including four failed placeholder queries, across the same four tasks. This version has not repaired tool query selection or grouped-statistic behavior. Selective facts still cannot correct routing stops where no first reply exists.

## A diagnostic soundness issue found during review

`dtype_vs_value_preservation` treats object-to-integer conversion as necessarily changing every cell. A synthetic input column explicitly stored as object but containing actual integers [1, 2] can be cast to integer dtype while preserving both values. That correct output passes the verifier, yet this diagnostic labels the contract as a proof-level contradiction.

The reproducer is saved in `audit.json` under `diagnostic_false_positive_probe`. It is not an observed regression on the submitted 100 tasks, and no current run is changed. It shows that passing the gold-contract screen is insufficient to establish universal soundness. Before broader use, verify actual cell values and overlapping preservation scope; unsupported analysis should produce evidence/uncertainty rather than a claimed proof.

## Recommendation

Keep selective-facts checked as a documented development variant: it adds two correct results without observed regression and uses fewer tokens. Tool checked currently demonstrates early stopping/cost savings, not greater task accuracy. Preserve both raw runs and report the unchanged wrong acceptance.

Prioritize a separately versioned development test of unresolved column-grounding behavior and repair of the diagnostic's overbroad dtype rule. Distinguish genuine changed, issue-resolving corrections from unchanged replies in reporting. Test more diverse legitimate empty filters, multi-edit tasks, casts and preservation scopes before making stronger hard-rejection rules. The pending 7B gold repair run can independently measure repair capability; these parser-check results do not answer that question.

Original uploads are preserved under `kaggle_outputs/stage2/contract_check_v1/` and `notebooks/executed/stage2/contract_check_v1/`. No inference source, contract, score, notebook output or adapter was modified during analysis.
