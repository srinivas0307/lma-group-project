# Baseline 2 on the current 3B stack

The full 100-task Baseline 2 run gives **75 correct accepted outputs, 11 incorrect acceptances and 14 withheld**. Outcomes are 86 ACCEPT, nine REJECT and five ESCALATE. The executed notebook matches its template and has no recorded exceptions; the output ZIP has a valid CRC, complete content seal and all 100 task records. CPU replay of all references, initial programs and recorded final programs finds **zero generation-input or scoring mismatches**.

## What system this run measures

Baseline 2 uses the request-only 3B FP16 generator, conventional data-quality checks with the fixed Stage 1 dev-tuned thresholds, and 3B FP16 repair with evidence/recovery features that require no contract. It loads no parser or adapter and makes zero parser calls. It has no intent contract, contract-derived repair hints or declared-schema guard.

The experiment name includes "hints" because it belongs to the matching current-stack baseline family, not because this conventional baseline receives contract hints. Some inherited common metadata still names the combined arm/guard. The actual system-specific repair version, per-task checker/arm/guard fields and source code consistently identify `evidence-no-contract`, conventional checks and no schema guard. This metadata distinction should be preserved when reporting the run.

The new source manifest is `7dfca3e3312c16718d1d4284ffa658698032697267a1d066de782ee5dcf036b4`, recorded in common.baselines_source_manifest_sha256; the plan's top-level source manifest refers to the embedded frozen contract-check pack. Both layers are retained for provenance. Inputs, task inventory, software/Python, generator and repair model revisions/precision, and budgets match the repair-hints experiment family. This is a whole-system comparison: checker, parser presence, generator conditioning and contract-dependent repair features differ intentionally.

## Why accuracy is lower than initial generation

There are **79 initially correct programs − five correct programs lost + one incorrect program repaired = 75 final correct results**.

The only successful repair is `adult_dedup_002`. The five lost initially correct tasks are `airbnb_filt_001`, `airbnb_filt_004`, `titanic_filt_003`, `titanic_filt_004` and `titanic_filt_005`. Each initial trace fails `distribution.mean_shift`. These are filtering requests: removing the requested rows legitimately changes column distributions, but a generic shift limit cannot distinguish an intended transformation from damage.

The CPU reference probe confirms the broader problem: **16 of 100 correct reference outputs are rejected**, all by distribution.mean_shift. Five recorded final candidates are also correct but withheld: airbnb_dedup_001, airbnb_dedup_002, airbnb_filt_001, titanic_dedup_004 and titanic_filt_005. Three of those began incorrectly and reached a correct final candidate, but it still could not be accepted under the generic checker. These outcomes remain withheld in the reported score.

The repair agent is therefore directed to satisfy a check that can conflict with the requested filtering. Some attempts introduce additional mistakes, such as deleting unknown-age passengers who should remain. Increasing repair capability alone does not make this check request-aware.

## Why incorrect programs are accepted

The 11 incorrect acceptances are `adult_conv_003`, `adult_dedup_003`, `adult_filt_005`, `airbnb_dedup_003`, `taxi_col_005`, `taxi_conv_001`, `taxi_dedup_002`, `taxi_dedup_005`, `taxi_filt_004`, `titanic_conv_001` and `titanic_conv_002`.

Examples of failures generic checks do not establish:

- `adult_conv_003` orders education categories alphabetically rather than by the requested education ranking.
- `titanic_conv_001` numerically encodes Sex rather than making it categorical.
- `taxi_filt_004` changes original datetime-column types and applies filtering conditions that do not match the requested transformation.

A candidate can preserve broad data-quality properties while implementing the wrong operation, key, order or condition. When the generic checker passes it, repair is not invoked to resolve the intent error. Accepted-output correctness is 75/86 = **87.2%**; the remaining 12.8% of accepted results are wrong on this sample.

## Comparison with current proposed systems

The sealed repair-hints results report:

- Proposed selective facts: **82 correct, two incorrect accepted, 16 withheld**. Relative to Baseline 2: seven more correct results and nine fewer incorrect acceptances. Baseline 2 alone solves seven tasks; selective facts alone solves 14.
- Proposed tools: **80 correct, one incorrect accepted, 19 withheld**. Relative to Baseline 2: five more correct results and ten fewer incorrect acceptances. Baseline 2 alone solves 11 tasks; tools alone solves 16.
- Gold contracts plus repair hints: **92 correct, zero incorrect accepted, eight withheld**. This is a supplied-contract oracle diagnostic, not a deployable learned-parser system.

This supports the current proposed systems over this conventional-check baseline on the observed tasks, with a cost/coverage trade-off. It is not evidence that every part of the architecture independently helps or that all baselines have been beaten.

## Cost and reporting limits

Baseline 2 uses **298,798 model tokens** and zero parser tokens. Current proposed selective facts uses 425,057 total tokens and tools uses 580,233. Baseline 2 is cheaper, while delivering fewer correct results and more incorrect acceptances. Summed recorded task time is approximately 10.8 minutes, excluding model loading/setup and reference scoring; this is not a controlled wall-clock benchmark.

The historical Baseline 2 result was 72 correct, 13 incorrect accepted and 15 withheld, using 7B NF4 repair and an older protocol. The current run gains three correct cases and loses none, but this is not an isolated repair-loop improvement experiment. In particular, historical direct generation should not automatically be treated as a matched current-stack direct baseline merely because it has no repair. The current run starts with 79 correct programs, while historical direct generation had 78. Only 67/100 initial programs match the historical direct trace byte for byte. A fresh matched direct run is needed for a clean current comparison.

Generic summary fields named reference_incompatible or parser_reference_verdict do not measure conventional-check validity: this baseline has no parser. The CPU audit explicitly applies the conventional checker to reference outputs to measure that instead.

These tasks have repeatedly informed development. Report results as diagnostic evidence, retain the fixed thresholds and record settings, and use fresh held-out evaluation for stronger research claims. Other current baseline notebooks are separate runs; this upload completes Baseline 2 only.

## Evidence and saved files

- `baseline2.json`: paired comparison with the recorded current proposed and historical Baseline 2 results.
- `baseline2_audit/audit.json`: source/seal/config checks, notebook inspection, input-prompt checks and CPU reference/initial/final execution checks, including any observed mismatches.
- Reproduction: `.venv/bin/python tools/audit_baseline2_hints.py`.
- Original bundle: `kaggle_outputs/stage2/baselines_hints_v1/stage2_baselines_hints_baseline2.zip`.
- Executed notebook: `notebooks/executed/stage2/baselines_hints_v1/stage2_baselines_hints_baseline2_kaggle_executed.ipynb`.

Original uploaded bytes are preserved and indexed. No inference code, thresholds, recorded outcomes or notebook outputs were modified during analysis.
