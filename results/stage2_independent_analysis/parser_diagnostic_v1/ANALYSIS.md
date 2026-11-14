# Parser diagnostic analysis

All four arms completed all 20 development tasks. Outer and nested ZIP seals, source/input hashes, adapter metadata, task coverage and supplied summaries validate. The executed notebook is the diagnostic despite its filename.
Notebook exception outputs: 0; source matches template: True. Reconstructed-message mismatches: 0; extraction mismatches: 0; CPU score mismatches: 0.

## Results

- original_nf4: 16/20 reference passes; 20/20 schema-valid contracts; 0/20 wrong no-ops accepted; 34076 tokens.
- profile_nf4: 8/20 reference passes; 19/20 schema-valid contracts; 0/20 wrong no-ops accepted; 65585 tokens.
- original_fp16: 15/20 reference passes; 20/20 schema-valid contracts; 0/20 wrong no-ops accepted; 33913 tokens.
- profile_fp16: 7/20 reference passes; 18/20 schema-valid contracts; 1/20 wrong no-ops accepted; 65700 tokens.

Reference pass means the generated contract accepts a correct reference output. This is not end-to-end Stage 2 accuracy or a complete semantic correctness score. There are 20 distinct tasks, evaluated four times.

## Interpretation

At fixed NF4 loading, adding the profile loses nine reference passes and gains one (16 to 8). At fixed FP16 loading it also loses nine and gains one (15 to 7). NF4 gives only one net additional pass within either input format. Quantization affects individual outputs, but restoring NF4 does not resolve the profile regression in this sample.
The profiles are derived from the input data and reconstructed exactly. Incorrect behavior is visible in generated contracts: wrong keys, literal/name changes, invalid functions and contradictory preservation. Input-format/distraction effects are a plausible explanation; this experiment does not isolate profile length, placement, contents or training mismatch separately.

## Concrete failures

- `taxi_col_006`: request renames Airport_fee to airport_fee; profile/NF4 instead demands filling the column with zero and increasing nulls. Original/NF4 accepts the correct rename.
- `taxi_dedup_006`: request specifies VendorID and pickup time, keeping last. Profile/NF4 uses all columns. Original/NF4 passes.
- `titanic_col_006`: request names passenger_class; profile/NF4 invents pclass and also preserves the removed original column. Original/NF4 passes.
- `airbnb_col_006`: profile/NF4 emits unsupported fn=rename and fails schema validation. Original/NF4 passes.
- `taxi_miss_006`: request marks zero trip distances missing; profile/FP16 accepts the wrong unchanged input. Profile/NF4 also fails the correct reference.

## What to do next

Do not treat the current full-profile v2 or untrained tool protocol as a validated improvement. Keep original-input NF4 as the current parser development control. Avoid spending on all six profile-based full runs before input-format work.
Use the existing 80-task synthetic development suite (after review) to test a compact, request-relevant information format against that control, including both tasks requiring statistics and tasks that do not. Supply only relevant columns/statistics and full precision; do not inject gold or reference-derived hints. If performance still regresses, train on the selected format with explicit scope/key/preservation examples and select using semantic tests.
These 20 development requests contain no median, mean, mode or group-statistic requests. Consequently this run demonstrates collateral damage on ordinary tasks, but does not test whether supplying genuinely unavailable statistics helps those harder tasks. A follow-up needs that coverage.
Zero wrong no-op acceptances in three arms does not establish strong safety: unchanged input is only one wrong-output probe. Use the broader wrong-candidate probes before selecting a parser. The synthetic suite remains draft, and sharing templates limits independent generalization claims.

The pip dependency-conflict messages are warnings printed in stream output; they did not stop these jobs. Torchao removal and the small LoRA probe succeeded. No GPU inference, retraining, implementation changes or original-result rewriting were performed by this audit.

Reproduce: `.venv/bin/python tools/analyze_parser_diagnostic.py`. Detailed cases, paired gains/losses, family counts, cost and provenance are in `audit.json`.
