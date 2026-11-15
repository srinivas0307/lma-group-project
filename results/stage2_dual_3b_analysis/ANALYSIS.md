# Stage 2 with unquantized 3B generation and repair

**Both independent methods finish with 73 correct accepted transformations, two incorrect accepted transformations and 25 withheld tasks. The tool method provides no net quality gain on these 100 tasks and uses 17.7% more total tokens. Prefer selective facts as the current lower-cost candidate, while investigating the shared false acceptances and remaining contract errors.**

## Completion, precision and comparability

Both notebooks complete their two-task smoke and full 100-task test jobs with no exception outputs. Full ZIP seals, task inventory, raw inputs, runtime source, source manifest, exact selected adapters and matching software environments pass the unmodified real-result collector.

The recorded actual generator and repair models are **Qwen2.5-3B-Instruct**, the same revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, with **no quantization** and floating weights **torch.float16**. The parser is the selected method-specific 1.5B adapter over NF4, as designed. No 7B model is loaded by either job.

Both runs use Python 3.13.16, Torch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0, pandas 2.3.3 and NumPy 2.1.3 on two Tesla T4 devices. The generator uses CUDA 1, parser/repair CUDA 0, with sequential residency. Generator prompt is `b2-v2`/`ed85f90bc61a`; repair prompt is `a1-v4`. The source manifest is `66d2d430b0f79400ea666c856023b3150af856b3fa420165448901754b0323a3`.

Thus the comparison has the intended equal generation/repair model strength. It still compares different parser adapters/input protocols, not tool access in isolation. These previously inspected 100 test tasks are diagnostic; they are not an independent confirmatory benchmark.

## End-to-end results

- **Selective facts:** 73 correct, two incorrect accepted, 25 withheld; 16 ESCALATE and nine REJECT.
- **Tools:** 73 correct, two incorrect accepted, 25 withheld; 16 ESCALATE and nine REJECT.

The equal totals are not duplicate runs. The contracts differ on 19 tasks, final correctness differs on six tasks, and ACCEPT/REJECT/ESCALATE differs on ten tasks. Both use the correct, distinct selected adapters.

Tools gains three tasks relative to selective facts:

- `airbnb_dedup_003`: it selects `host_id` rather than all columns.
- `taxi_col_005`: it avoids preserving a newly introduced column as if it existed in the input and repairs the zero-distance ratio case.
- `taxi_miss_005`: it generates the sentinel-to-null contract; selective facts stops in the router before inference.

Tools loses three tasks relative to selective facts:

- `taxi_conv_002`: it demands datetime conversion while preserving the original timestamp strings unchanged.
- `taxi_miss_003`: replacing passenger count zero with one is paired with an impossible requirement that null count increase.
- `taxi_miss_004`: the same contradictory null-increase requirement accompanies replacement of RatecodeID 99 with one.

No operator normalization is applied in either full test run: `normalized_tasks=0`. This does not invalidate the earlier development gain from normalization; the earlier 88/100 result was on 80 synthetic plus 20 original development cases, not these 100 end-to-end test tasks.

## Parsing versus full-system performance

Selective facts emits 95 schema-valid contracts: 83 accept the correct reference, nine violate it, and three give UNCERTAIN. Five tasks have no usable contract: two facts-router stops, two invalid contracts and one invalid JSON answer.

Tools emits 99 schema-valid contracts: 84 accept the correct reference, 12 violate it, and three give UNCERTAIN. One task stops after invalid parser responses.

Therefore fewer syntax/parse failures do not translate into a material reference-compatibility gain: tools has four extra valid contracts but only one extra reference pass, and three more reference-incompatible contracts overall (15 versus 12). A valid contract can still encode the wrong request or an impossible preservation rule.

The earlier 88/100 selective-facts and 90/100 trained-tool numbers are parser-development checks. They measure different tasks and score correct reference transformations directly. The current 73/100 requires the full generator/verifier/repair pipeline to produce a correct accepted transformation. They are not interchangeable accuracy figures.

## Why 25 tasks are withheld

For selective facts, withholding decomposes into:

- **Five** tasks stopped at parsing.
- **Eleven** tasks with contracts that do not accept the correct reference.
- **Nine** tasks whose contracts accept the correct reference, but generation/repair does not produce a correct accepted result.

For tools:

- **One** task stopped at parsing.
- **Fourteen** tasks with reference-incompatible contracts.
- **Ten** tasks with reference-compatible contracts but unsuccessful generation/repair.

The runtime initially generates 76 correct outputs for selective facts; it withholds five of them and repairs two initially incorrect outputs: **76 − 5 + 2 = 73**.

Tools initially generates 77 correct outputs, withholds nine and repairs five: **77 − 9 + 5 = 73**. It repairs more cases but loses more initially correct ones to incorrect contracts. It therefore provides no net correctness gain.

Repeated identical code proposals are frequent in the repair traces. Reference-compatible failures on deduplication/column/filter tasks also show that better parsing alone will not solve every case. Conversely, a correct repair cannot satisfy a logically wrong contract and preserve the actual request simultaneously; not every repair failure demonstrates insufficient model capability.

## The two shared incorrect acceptances

### Airbnb location filter: wrong intent column

`airbnb_filt_002` asks to keep listings located in Brooklyn and Manhattan. Both parsers choose `neighbourhood`, while the input's borough values are in `neighbourhood_group`. Both generators then filter using that wrong column.

The resulting empty output satisfies the wrong predicted contract, but the correct reference retains 927 rows. This is a semantic intent/schema-resolution failure, not an inability to execute the filter or a GPU error. Input category-value inspection could help disambiguate the columns; the tool parser makes no query for this task. The benchmark's gold assumption should not be injected as a post-hoc hint into ordinary inference.

### Taxi date filter: extra helper column not forbidden

`taxi_filt_004` correctly filters pickup dates, but the generated final program leaves an extra `pickup_datetime` helper column in the output. The reference returns 997 rows with the original schema.

The predicted contract accepts the wrong output despite also accepting the reference. The existing `columns_unchanged` predicate checks that the named input columns remain with unchanged values/dtype families; it does not forbid undeclared extra output columns. The benchmark gold contract uses the same predicate, so a successful gold-contract test alone would not rule out this gap.

This is a shared **schema-preservation verification gap**, not just an incorrect parser literal. A separately versioned guard against undeclared column additions, with tests for legitimate add/rename operations, is a concrete inference/verifier improvement. The current experiment has not been silently rescored or modified to include it. The generator prompt already asks to remove temporary columns, but prompt instructions alone did not prevent this case.

## Facts routing and tool use

Selective facts supplies four real full-column facts: native-country mode, reviews-per-month median, Embarked mode and Age median. Their values are computed from the CSV, not reference outputs.

The tools parser makes **eight attempts across four tasks**: four `__unknown_column__` failures and four successful `profile_column` executions. This is the same recovery-prefix artifact seen in development, now on four test tasks. These are parser queries, distinct from the repair agent's own tool calls.

Three successful queries retrieve relevant statistics (reviews_per_month, Embarked and Age). The fourth, on `titanic_miss_005`, queries **Pclass** instead of **Age grouped by Pclass**. It then proposes filling all missing ages with 3. The correct input-derived grouped medians are 37, 29 and 24, and the generated correct grouped-median code is rejected against that wrong literal.

An audit-only call to the already available `group_profile(column="Age", by=["Pclass"])` returns the needed values. This observation was not requested by the model in the submitted run; it is diagnostic evidence that the tool can retrieve the information, not evidence of successful model-selected retrieval.

Two selective-router stops deserve separate treatment:

- **`taxi_miss_005` is a routing false positive.** Its request says the sentinel values “both **mean** 'not recorded'”. The lexical router misreads `mean` (signify) as a statistic, and then demands a conditional statistic scope. This is an implementation weakness in the conservative router, not a missing statistic or a necessary escalation. A CPU-only diagnostic changing that neutral verb to `denote` yields no requested facts, confirming the ambiguity. No new prediction or replacement score was produced from that rephrasing.
- **`titanic_miss_005` needs a full per-group statistic map.** The implemented selective grammar supports a single explicit equality group, not “each row's same class” over all groups. It stops explicitly. The tool version runs but uses the wrong query/contract, so tool availability alone does not solve the grouped request.

These observations narrow the next information-handling work: distinguish a statistic request from ordinary language and support/test genuine grouped statistics. They do not justify retraining merely because a case failed before reaching the model.

## Cost and family results

Selective facts uses **177,514 parser tokens**, 98 parser calls and **537,681 total tokens**. Tools uses **258,796 parser tokens**, 110 parser calls and **632,678 total tokens**. Tools adds **94,997 total tokens (+17.7%)**, with parser cost **45.8% higher**, for no net end-to-end quality gain.

The repair agent makes 104 model calls / 49 tool calls for selective facts, versus 110 / 46 for tools. These repair tools are not the eight parser-tool attempts. Summed recorded per-task durations are approximately **32.0 minutes** and **37.1 minutes**; these exclude setup, model loading and reference-scoring overhead, so they are not full notebook wall-clock times.

Correct results out of 20 per transformation family:

- Column transformations: selective **14**, tools **15**.
- Type conversions: selective **16**, tools **15**.
- Deduplication: selective **12**, tools **13**.
- Filtering: both **14**, with both incorrect acceptances in this family.
- Missing-value tasks: selective **17**, tools **16**.

## Recommendation

Keep selective facts as the main lower-cost candidate on current evidence. Retain tools as a separately reported method: it improves some contracts and reduces parsing stops, but has not improved final quality here. Do not label tools superior from its earlier development score or its number of tool calls.

The next investigation should prioritize the shared semantic/schema-preservation failures and the confirmed router false positive, with development regression checks in a new version rather than silently changing this completed experiment. No further training is required merely to test those bounded implementation changes.

Before claiming an architectural advantage, run a matching direct-3B baseline and gold-contract diagnostic with this same 3B FP16 generation/repair software and prompt/version settings. The old stored 78-correct direct baseline is historical context, not a clean comparison to this new pipeline. The gold arm can estimate parser-related loss but must also be interpreted alongside the schema gap above. Expand to the remaining six-system presets when the comparison is useful; do not spend on a larger benchmark simply to hide known failures. A truly independent benchmark remains necessary for a stronger eventual research claim.

## Verification and files

The unmodified strict collector has been run locally from the exact sealed source package; its report is in `comparison/summary.json`, `comparison/SUMMARY.md` and `comparison.zip`. It confirms model-quality bundles, matching inputs, source versions, adapters, precision and environment.

`tools/analyze_stage2_dual_3b.py` reproduces the collection and CPU audit. It reconstructs parser/control flow, verifies tool observations and generation prompts, reexecutes reference programs, initially generated code and accepted final code, and compares recorded correctness and verifier labels. Local Python differs from Kaggle; any replay differences are retained in `audit.json`. Candidate outputs are cached in `/tmp` by bundle/runtime/input-bound fingerprints to avoid redundant reruns of identical code. No GPU inference, retraining or runtime modification is performed.

The uploaded artifacts are organized under:

- `kaggle_outputs/stage2/dual_parser_3b_fp16_v1/` for the two full result ZIPs.
- `notebooks/executed/stage2/dual_parser_3b_fp16_v1/` for the two executed Kaggle notebooks with descriptive names.

Original uploaded bytes are preserved. This report's numerical conclusions come from the submitted run records; audit completion/mismatch details are recorded separately in `audit.json`.

Completed CPU audit: all 200 task records passed parser/control-flow, generated-prompt, reference-verdict, initial-code and accepted-final-code replay checks with **zero policy, generation or score mismatches**. Both executed notebook sources match the published Kaggle templates and have zero exception outputs. Both incorrect final outputs and their predicted/gold verdicts are independently reproduced: Airbnb returns zero rather than 927 rows; taxi returns 997 rows plus the undeclared helper column, which also passes the existing gold contract. The bounded routing/group diagnostics are reproducible in `input_diagnostics.json`; they do not alter submitted predictions or scores.

## Follow-up failure review and improvement plan

A complete review of the 54 unsuccessful method/task outcomes across 30 distinct tasks is in [the failure inventory](failure_review/INVENTORY.md). The [3B improvement plan](../../docs/stage2_3b_improvement_plan.md) distinguishes repair evidence, contract diagnostics, selective-facts/tool routing and schema verification. These are proposed changes; the frozen runtime and scores remain unchanged.
