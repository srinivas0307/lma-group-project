# Selective facts without contract conditioning of generation

The complete 100-task run gives **79 correct, two incorrect accepted and 19 withheld**. The matched proposed selective-facts repair-hints system gives **82 correct, two incorrect accepted and 16 withheld**. Omitting the contract from generation saves 12.1% of model tokens but loses three correct completions overall.

## Configuration and validation

This is a generator-contract ablation, not a parser-free or contract-free system. The selected 1.5B NF4 selective-facts parser still generates contracts, contract-check-v1 still applies, and 3B FP16 verification/repair retains the contract, schema guard and repair hints. The initial 3B FP16 generator receives the request and input schema/sample without the contract.

The ZIP has a valid CRC, complete SHA-256 seal, successful completion marker and all 100 task records. The executed notebook code matches the template and has no recorded exception outputs. Models, adapter hashes, inputs, task inventory, environment, parser settings, thresholds, repair prompt and budgets match the proposed selective-facts run. All 100 saved parser/correction records match after excluding timing and backend-path metadata.

`tools/audit_no_contract_tools_hints.py --method selective` verifies the source package and runtime, then replays reference, initial and accepted-final outputs on CPU. All 100 task records passed the CPU audit with **zero generation-input, reference-verdict or correctness mismatches**. Evidence is saved in `no_contract_selective_audit/audit.json`. Local Python is 3.12 versus recorded Kaggle Python 3.13.16; agreement covers these observed cases only.

## Gains, losses and failure accounting

The ablation alone solves `airbnb_dedup_005`. The proposed version alone solves `taxi_dedup_002`, `taxi_dedup_005`, `titanic_dedup_001` and `titanic_dedup_005`. Four gains minus one loss favor supplying the contract to generation by three tasks.

The ablation has **75 initially correct − four lost + eight repaired = 79**. Proposed selective facts has **74 − one lost + nine repaired = 82**. A slightly higher initial correctness without contract conditioning does not translate into better final accepted correctness: acceptance alignment and repair trajectories differ.

The same two incorrect acceptances remain: `airbnb_dedup_004` and `airbnb_filt_002`. The latter starts correctly but is changed by repair to satisfy the wrong neighbourhood-column contract. Removing the contract from generation cannot remove errors caused by its later use in verification and repair.

Initially correct outputs lost are `adult_col_005`, `adult_miss_005`, `airbnb_filt_002` and `titanic_col_005`. This metric includes the wrong acceptance above, rather than meaning four additional withheld tasks. Ten parse stops and five emitted reference-incompatible contracts remain unchanged because the parser outputs are unchanged.

## Cost and interpretation

Total model tokens decrease **425,057 → 373,717**, saving **51,340 (12.1%)**. Parser tokens remain 208,324; generation/repair tokens decrease 216,733 → 165,393. Recorded summed task time is approximately 32.5 minutes versus 34.4 minutes proposed, excluding setup/loading and reference scoring; this is not a controlled throughput benchmark.

Together with the tools ablation (78/1/21 versus proposed 80/1/19), this supports supplying predicted contracts to generation under the current protocol, with higher token cost and task-specific regressions. It does not establish that contract conditioning always helps, eliminate contract-quality failures, or constitute independent evaluation: these test cases have repeatedly informed development.

Retain this run as the selective-facts generator-contract ablation in the report. Keep historical no-contract runs separate because their parser, repair model and protocol differ. Baseline 3 and a freshly matched direct baseline remain separate work.

Saved original uploads:

- `kaggle_outputs/stage2/baselines_hints_v1/stage2_baselines_hints_no_contract_selective.zip`
- `notebooks/executed/stage2/baselines_hints_v1/stage2_baselines_hints_no_contract_selective_kaggle_executed.ipynb`

Paired task results are in `no_contract_selective.json`. Original bytes are preserved and indexed; no model, contract, threshold, score or notebook output was modified during analysis.
