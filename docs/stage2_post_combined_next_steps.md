# Next step after the combined 3B runs

Status: proposed development experiment, not implemented or GPU-validated.

The [completed analysis](../results/stage2_combined_3b_analysis/ANALYSIS.md) reports 74 correct, one incorrect accepted and 25 withheld for each parser method. Original 3B runs gave 73/2/25. Keep these completed experiments and sources frozen.

## What to address

The combined repair loop helps slightly, but cannot reliably satisfy a contract that contradicts the requested transformation. Selective facts loses five initially correct generated outputs; tools loses nine. Among withheld tasks, selective facts has five parse stops, 11 reference-incompatible contracts and nine reference-compatible unsuccessful repairs. Tools has one, 14 and ten respectively. These are audit categories, not guaranteed recoverable gains.

Both methods still accept a wrong filter because a valid column name has the wrong meaning. An output-schema guard cannot resolve semantic column selection. A zero-row result alone is not an error: legitimate empty results must remain supported.

## Proposed separate development version

Keep the selected 1.5B parser adapters, NF4 parser precision, 3B FP16 generation/repair, combined repair loop and existing proposal limits fixed. Do not retrain or change the existing notebooks. Develop a new parser protocol with the following separately measured changes:

1. **Selective information routing.** Recognize requests for statistics without treating ordinary uses of “mean” as statistical instructions. Handle explicit grouped statistics with typed group keys, full precision and explicit unavailable/ambiguous outcomes. No global-statistic substitution for a group statistic. Unrelated requests receive no statistical profile.
2. **Grounded diagnostics.** Reuse the existing unknown-column and change/preservation checks. Extend only with demonstrable inconsistencies, such as incompatible fill and null-count requirements on the same guaranteed-retained rows. Treat competing column/category matches as evidence for clarification or correction, not proof permitting automatic column replacement. Bound evidence and label its source.
3. **One evidence-led correction.** Only when a diagnostic fires, show the request, contract and relevant input evidence to the existing adapter. Normalize and validate the reply again. A remaining proven contradiction is withheld with a reason. Do not silently delete constraints. Earlier generic retries gave no gain, so test this arm separately rather than assuming it helps.
4. **Tool-specific instructions.** Clarify value column versus grouping keys, require exact input column names and provide useful errors for placeholders. Preserve model-selected queries. Automatically computed facts would constitute a separate hybrid method, not an improvement attributed to tool selection.

Live diagnostics may use the request, input data and predicted contract only. They must never access reference code, reference outputs, gold contracts or evaluation labels.

## Development comparison

Start with new synthetic development cases using different names, values and wording. Include ordinary non-statistical language, grouped fills, all-missing groups, missing group keys, similar columns, legitimate empty filters, ambiguous category matches and valid preservation conditions. Review gold/reference pairs and targeted wrong outputs before running a model. These are development cases informed by observed failures, not an independent benchmark.

Use paired arms per method: current parser control; routing/instruction changes only; routing/instructions plus diagnostics and at most one correction. Keep all other settings fixed. Record original and corrected contracts, diagnostics, facts/tool observations, all calls/tokens and per-case changes. Evaluate whether correct outputs pass and whether targeted wrong outputs are accepted; reference compatibility alone is insufficient.

Advance a variant only if it improves correct-output compatibility without increasing observed wrong-output acceptance on the same development suite. Report regressions, ambiguity/withholding and extra cost. Then check end-to-end development behavior before another full Stage 2 run. This is a screening rule, not a guarantee of generalization.

## What to do now

No rerun of the current full notebooks is needed to complete their analysis. The next implementation deliverable is a versioned parser-development package with CPU tests and a paired Kaggle notebook using the existing adapters. Only a demonstrated development improvement should be integrated into new end-to-end notebooks for both methods.

For a report now, use the completed combined results with their modest gain, token increase and remaining errors. The historical direct baseline comes from an older matched experiment; do not present it as a fresh controlled comparison with this changed pipeline. Stronger claims require matched baselines and a fresh held-out evaluation after the method is frozen.
