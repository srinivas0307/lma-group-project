# Tool-enabled Stage 2 proposed run analysis

**Completed, but the intent parser made zero tool calls.** The measured result is 67 correct, 1 incorrect accepted and 32 withheld. This is not evidence that parser tool use improved results.

## Validation

All 100 task records and 100 fresh parser records pass the strict collector and ZIP CRC check. Notebook exception outputs: 0; source matches the tool template: True. Expected adapter hashes match.
Exact prompt/control-flow/counter checks and CPU replays were run without model inference. Any mismatches are listed in audit.json.
- Input prompt mismatches: 0
- Parser control-flow mismatches: 0
- Recorded parser-counter mismatches: 0
- Reference-verdict replay mismatches: 0
- Initial/accepted-final correctness replay mismatches: 0

## What the parser actually did

The parser made 111 model calls: 89 tasks returned a schema-valid contract immediately; 11 returned an invalid response and then chose explicit escalation after the correction message. All 100 saved parser traces contain zero tool attempts and zero observations. No assistant response is a tool-call JSON object. Zero tool failures here means zero attempts, not proven tool reliability.
The repair agent separately made 67 tool calls. Those tools inspect/repair generated code after parsing and are not the parser's profile_column, count_rows, inspect_rows or group_profile tools.
The prompt lists the read-only dataset tools and instructs the model to query missing facts. The existing adapter was trained to emit a final contract, not tool conversations. The runtime accepts a schema-valid final contract without enforcing a prior query. The observed behavior is consistent with that training/protocol mismatch; this run does not isolate its causal components.

## Outcomes and withholding

The parser produced 89 valid contracts and 11 explicit escalations. Of the valid contracts, 71 accept the correct reference and 18 reject it. The 32 withheld tasks comprise 11 parser escalations, 17 reference-incompatible contracts, and 4 reference-compatible tasks whose generation/repair did not produce a correct accepted result.
The generator initially produced 71 correct outputs. Twelve were ultimately withheld and eight initially incorrect outputs were repaired: **71 - 12 + 8 = 67**.

## Direct examples

- `titanic_miss_004`: asks for median Age. The first parser reply guesses 39.0 and uses unsupported `<>`; after feedback it escalates. It never queries the tool. An audit-only query returns the true input median 28.0.
- `titanic_miss_002`: asks for modal Embarked. The parser chooses C without querying. The actual mode is S (644 occurrences). Correct generated code is withheld against this wrong literal.
- `taxi_dedup_001`: the parser selects the correct composite key, but adds invalid null_count op=not_null and preserves row count. It escalates rather than correcting the contract; tools would not automatically solve these semantic/format errors.
- `airbnb_filt_002`: the sole incorrect acceptance. The parser checks neighbourhood instead of neighbourhood_group. Generated code satisfies that wrong contract while removing every row; the reference retains 927. The exact code and replay verdicts are saved in audit.json.
The two tool-query outputs in audit.json were calculated locally for explanation. They were not observations made by the model during the submitted run.

## Comparisons and limits

This tool-protocol run gives 67/1/32, versus the earlier frozen proposed system's 76/2/22 and the full-profile FP16 run's 44/2/54. It gains 23 correct tasks in net over the full-profile run, but no parser tools were used: that gain cannot be credited to tool retrieval. It remains nine correct tasks below the original proposed result.
The tool run uses the original adapter over an FP16 base, the newer package environment, new tool/escalation instructions and correction behavior. The frozen original parser used NF4; these are diagnostic experiments, not a clean tool/no-tool ablation. Model revisions, per-task gains/losses and family results are recorded in audit.json. Previously inspected test tasks are not fresh confirmatory evidence.

## Transformation families

Correct accepted outputs out of 20 per family: column transformations 18, type conversions 15, filtering 15 (plus the single incorrect acceptance), missing-value tasks 11, deduplication 8. Details and parser escalation counts are in audit.json.

## Cost

Parser: 241,532 tokens across 111 calls. Generation/repair: 438,914 tokens. Total: 680,446. Summed recorded parser/runtime duration is 34.7 minutes; it excludes setup, model loading, reference scoring and the smoke job.

## Recommended next action

Keep the current selective-facts training running. That model learns to use automatically supplied facts; it does not train tool selection. This untrained tool-protocol experiment is separate and does not invalidate the current training run.
For genuine model-selected tool behavior, a later adapter needs matching multi-turn examples that supervise assistant tool requests, tool observations and final contracts. Evaluate tool uptake specifically on tasks requiring unavailable facts, observation-to-literal correctness, scope correctness, cost and wrong-output acceptance. A runtime requirement to retrieve missing facts can prevent guessing, but forcing retrieval in code is orchestrated/hybrid retrieval and must be reported as such.
Do not attach the eventual sf1 adapter directly to this old tool notebook: its protocol and strict expected adapter hashes differ. Choose and validate the input protocol before another tool-training experiment.

Reproduce: `.venv/bin/python tools/analyze_stage2_tools.py`. No GPU calls, adapter changes or runtime/source modifications were made by this audit.
