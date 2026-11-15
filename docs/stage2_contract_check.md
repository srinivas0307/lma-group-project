# Contract checks with one correction (contract-check-v1)

Status: completed and audited GPU runs. Selective facts gives **76 correct / 1 incorrect accepted / 23 withheld**, while tools gives **74/1/25**. Both controls were 74/1/25. All 200 records pass CPU policy/score audit with zero mismatches. See [the completed analysis](../results/contract_check_v1/completed/ANALYSIS.md), including the unchanged wrong acceptance and a demonstrated diagnostic false-positive limitation. Phase 1 of the improvement plan that followed the gold-contract run (gold 84/0/16 versus 74/1/25 for each parser method).

## What changes

Only the parsing step. After the unchanged first parse, input-only checks (`eval/stage2_contract_check/diagnostics.py`) inspect the contract using the request, the input data and the contract itself. Reference code, reference outputs and gold contracts are never visible. Each issue is either:

- **proof**: the contract contradicts the input data (for example a removal that matches no rows while the row count must decrease, deduplication on a key without duplicates, a type conversion while the same cells must keep their values, a null-count requirement contradicted by the contract's own fills).
- **evidence**: suspicious but possibly legitimate (a filter value absent from its column but present in another column, mapping keys that never occur while other values stay unmapped, a change predicate that changes nothing).

If any issue fires, the same parser gets one correction turn listing the issues (`correction.py`). Decision rule, fixed in advance: use the corrected contract if it parses, has no proof issue and no more issues than the first; otherwise keep the first contract if it only had evidence issues; otherwise withhold at the parse stage with the remaining proof issues as the reason. Generation, combined repair, schema guard, models, adapters, prompts and limits are identical to the completed combined 3B runs, so a corrected contract also reaches the contract-aware generator.

## Offline screen on saved contracts

`tools/analyze_contract_diagnostics.py` runs the checks on the 300 contracts saved by the completed runs and scores them against reference verdicts afterwards ([results](../results/contract_check_v1/offline_screen.json)):

| Contracts | Flagged |
|---|---|
| Selective facts, reference-incompatible | 11 / 12 |
| Tools, reference-incompatible | 13 / 15 |
| Selective/tools, reference-compatible | 0 / 83, 0 / 84 |
| Gold | 0 / 100 |
| Shared wrong acceptance `airbnb_filt_002` | flagged in both methods |

No proof issue fires on any contract that led to a correct final result. **These checks were designed after inspecting failures on these same test tasks**, so this screen measures coverage of known mechanisms, not generalization. It also does not show whether the 1.5B parser can use the feedback: an earlier generic one-correction experiment gave no gain. Unit tests use synthetic data with unrelated names, and all 120 dev/test gold contracts must raise no proof issue.

## Running it

1. `.venv/bin/python -m notebooks.kaggle.stage2_contract_check.make_notebooks` (already built: `notebooks/kaggle/stage2_contract_check/`).
2. On Kaggle, per method: attach `stage2_contract_check_source.zip` and the same adapter dataset as the completed combined run, enable Internet, T4 x2, Save Version → Run All.
3. Download `stage2_contract_check_<method>_checked.zip` and run
   `.venv/bin/python tools/analyze_contract_check_run.py --method <method> --checked <zip>`.
   It reports paired gains/losses versus the frozen control, decisions, extra parser tokens, and whether the first parse reproduced the control contract.

Tests: `tests/test_stage2_contract_check.py`.

## Limits

Selective-facts routing failures with no parser reply (`taxi_miss_005`) cannot be corrected by this step. Semantically wrong but consistent contracts (for example `airbnb_dedup_004`) are not detected. Withholding a proven-inconsistent contract trades a likely rejection for lower cost; it does not by itself add correct results.
