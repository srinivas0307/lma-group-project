# Tool-use parser with no contract supplied to generation

The completed run achieves **78 correct, one incorrect accepted and 21 withheld** on 100 tasks. The matching proposed tool-use repair-hints system gives **80 correct, one incorrect accepted and 19 withheld**. Omitting the contract from generation saves 13.5% of total tokens but loses two correct completions overall.

## What this ablation means

"No contract" applies only to the initial code generator. The trained 1.5B NF4 tool parser still generates a contract; contract-check-v1 still checks it and may request one correction; verification and the 3B FP16 repair loop still use the resulting contract, schema guard and repair hints. Generation receives the request and input schema/sample, without contract text. This is neither a direct baseline nor a tool-free system.

The recorded models, adapter hashes, input hashes, task inventory, Python/software environment, parser settings, thresholds, repair prompt and budgets match the current proposed tools run. The 100 saved parser/correction records match after excluding timing and backend path labels. The uploaded adapter comes from a different Kaggle account/dataset path, but its validated hashes match the selected adapter; this is not a different trained model.

The bundle has a valid ZIP CRC and complete SHA-256 seal, contains all 100 records and declares successful completion. The notebook code matches the published template after normalizing notebook source representation, and has no recorded exception outputs. The source pack and runtime hashes are verified by `tools/audit_no_contract_tools_hints.py`; CPU replay validates all 100 task records, including generation inputs, reference verdicts, initial correctness and accepted final outputs, with **zero mismatches**; details are saved in `no_contract_tools_audit/audit.json`. Original uploads remain unchanged.

## Paired results

The no-contract run alone solves three tasks: `airbnb_conv_001`, `airbnb_dedup_001`, `airbnb_dedup_005`.

The proposed tool system alone solves five: `taxi_conv_001`, `taxi_dedup_002`, `taxi_dedup_005`, `titanic_dedup_001`, `titanic_dedup_005`.

Thus five gains minus three losses favor supplying the contract to generation by two tasks on this sample. This supports contract conditioning in this experiment, but its benefit is not universal or independent held-out evidence.

No-contract initial correctness is 73, five initially correct programs fail to finish correctly, and ten initially incorrect programs become correctly accepted: **73 − 5 + 10 = 78**. Proposed tools has **70 − 2 + 12 = 80**. Better aggregate initial correctness without the contract does not guarantee better final completion; acceptance alignment and repair trajectories matter.

Eight tasks stop during parsing in both configurations. Their reference-incompatible emitted contract count remains seven because the contracts are unchanged. This ablation does not improve parser reliability or tool-query behavior.

## A concrete contract-induced failure

`airbnb_filt_002` is initially implemented correctly using neighbourhood_group to retain Brooklyn and Manhattan. The predicted contract instead uses neighbourhood. Verification rejects the correct candidate against that wrong contract, and repair changes the filter to the wrong column. The final incorrect result is accepted.

The error starts in the contract. The verifier enforces it, and repair follows that feedback. Hiding the contract from generation therefore cannot remove damage later caused by a wrong contract. This is a useful causal trace within this task, not evidence that all failures have the same cause.

The other initially correct programs lost are `adult_col_005`, `titanic_col_005`, `titanic_conv_003` and `titanic_miss_005`. The five-lost metric includes the incorrect acceptance above; it does not mean five additional withheld tasks.

## Cost, comparison limits and recommendation

Total tokens fall from **580,233 to 501,892**, saving 78,341 (**13.5%**). Parser tokens remain exactly 298,416; generation/repair tokens fall from 281,817 to 203,476. Summed recorded task time falls from approximately 37.0 to 35.3 minutes, excluding loading/setup and reference scoring; this is not a controlled wall-clock benchmark.

Historical no-contract results were 73/2/25, with a different repair/model/parser protocol. Keep those separate. The matched comparison here is the current proposed tool-use result at 80/1/19. All these repeatedly inspected tasks provide development diagnostics, not independent confirmation.

Retain this run as the generator-contract ablation for the report. The proposed version delivers two more correct outputs with the same observed incorrect-acceptance count, at higher token cost. Continue addressing unresolved contract grounding and admission; removing generator conditioning alone does not solve wrong-contract checking and repair.

Saved files:

- `kaggle_outputs/stage2/baselines_hints_v1/stage2_baselines_hints_no_contract_tools.zip`
- `notebooks/executed/stage2/baselines_hints_v1/stage2_baselines_hints_no_contract_tools_kaggle_executed.ipynb`
- `results/baselines_hints_v1/no_contract_tools.json` (paired scores/task IDs).
- `results/baselines_hints_v1/no_contract_tools_audit/audit.json` (CPU replay and integrity checks).
