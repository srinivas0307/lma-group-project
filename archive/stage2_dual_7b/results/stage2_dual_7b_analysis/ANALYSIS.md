# Stage 2: 7B NF4 repair compared with 3B FP16 repair

Both 7B-repair methods complete 100 tasks with **78 correct accepted outputs, one incorrect accepted output and 21 withheld outputs**. Each gains five correct outputs over its corresponding 3B FP16 repair run (73 correct, two incorrect accepted, 25 withheld). The tool parser offers no net accuracy gain over selective facts and uses 34.0% more total tokens in the 7B runs.

## What actually ran

The generator is **Qwen2.5-3B-Instruct, unquantized FP16**. The repair agent is **Qwen2.5-7B-Instruct, NF4**. The parser is the existing selected method-specific **1.5B adapter over NF4**. These notebooks do not use 7B as the initial generator, and they do not train an adapter.

Both full bundles are real, non-smoke runs of `stage2-dual-parser-v1`. The strict collector checks seals, source, adapters, task coverage and configurations. The uploaded notebooks have no exception outputs. The selective notebook matches the current template exactly. The tools notebook differs only in upload guidance and file-discovery print messages; its execution cells otherwise match, and its sealed runtime/source checks pass. Runtime files match the sealed source package. The old backend records resolved model identity and declared quantization with load guards; it does not provide the richer actual parameter-dtype metadata present in the newer 3B workflow.

Corresponding 3B and 7B runs use the same task/input hashes, adapters, parser settings, generator/repair prompts, limits and thresholds. Parser payloads are identical after excluding elapsed time and the backend's model label; generator payloads are identical after excluding elapsed time. Initial code matches on all 100 tasks for each method. This establishes a repair-stage comparison with the same starting outputs. It changes **both repair size and precision**, so it is not a size-only experiment.

Both methods use three repair attempts, eight tool calls per attempt, a 60,000-token task budget, generator output cap 512 and repair output cap 768. Generator prompt is `b2-v2`, repair prompt `a1-v4`. Environment: Python 3.13.16, Torch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, bitsandbytes 0.50.2, pandas 2.3.3, NumPy 2.1.3, two Tesla T4 GPUs.

## Results and repair contribution

- Selective facts: 78 correct, one incorrect accepted, 21 withheld (13 ESCALATE, eight REJECT). Initially 76 outputs are correct; five correct outputs are lost to the acceptance process, and seven initially incorrect outputs are repaired: **76 − 5 + 7 = 78**.
- Tools: 78 correct, one incorrect accepted, 21 withheld (nine ESCALATE, 12 REJECT). Initially 77 outputs are correct; nine correct outputs are lost, and ten initially incorrect outputs are repaired: **77 − 9 + 10 = 78**.

In the corresponding 3B runs, selective facts repairs two and tools repairs five initially incorrect outputs. The 7B agent increases both repair counts by five. It does not reduce the number of initially correct outputs withheld because of contract or verification problems.

For each method the same six tasks improve relative to 3B: `adult_dedup_003`, `airbnb_conv_005`, `taxi_col_004`, `titanic_col_001`, `titanic_col_003`, `titanic_dedup_004`. One task regresses: `airbnb_dedup_005` succeeds with 3B but escalates with 7B after repeated identical code proposals. Thus six gains minus one loss produces the five-task net improvement; a larger model is not universally better.

The decrease in incorrect acceptance comes from `taxi_filt_004`. The 3B agent accepts a date-filter result with an undeclared helper column. The 7B agent instead changes the original timestamp column's dtype and repeatedly fails preservation verification, so it REJECTs. This is safer withholding, **not a successful repair**. The verifier's permissive treatment of extra columns is unchanged.

## Remaining incorrect acceptance

`airbnb_filt_002` asks for Brooklyn and Manhattan listings. Both parsers encode a filter on `neighbourhood`, whereas borough values occur in `neighbourhood_group`. The 7B agent produces code satisfying that wrong contract, returns zero rows rather than the reference's 927, and is accepted. The reference verdict for that output is VIOLATION.

This is a semantic column-resolution failure. Increasing repair capacity does not fix an incorrect intent contract. Neither parser requests category-value evidence for this task. A separately evaluated development improvement could retrieve relevant column values and resolve the ambiguity before contract generation; hard-coding the benchmark's expected column would not demonstrate generalization.

## Why 21 outputs are withheld

Selective facts has five parsing/facts-routing stops, 11 withheld cases with reference-incompatible contracts, and five with reference-compatible contracts but unsuccessful generation/repair.

Tools has one parsing stop, 14 withheld cases with reference-incompatible contracts, and six with reference-compatible contracts but unsuccessful generation/repair.

The compatibility check here is an audit against the correct reference, not feedback given to the running parser or agent. A reference PASS does not prove the contract fully captures intent, as the extra-column gap shows. Conversely, failure to accept the correct reference indicates a blocking contract/verifier mismatch.

The known selective-facts router problems remain: `taxi_miss_005` interprets the ordinary word “mean” in “both mean 'not recorded'” as a statistical request; `titanic_miss_005` needs grouped Age medians beyond the current router's supported scope. The tool parser queries Pclass instead of Age grouped by Pclass on the latter task and emits the wrong fill value. A stronger repair model cannot consistently satisfy the request and a conflicting contract simultaneously.

## Selective facts versus tools

Equal totals hide different cases. Tools alone succeeds on `airbnb_dedup_003`, `taxi_col_005`, `taxi_miss_005`; selective facts alone succeeds on `taxi_conv_002`, `taxi_miss_003`, `taxi_miss_004`. The tools method still creates contradictory preservation/null-count conditions on some tasks.

Correct counts out of 20 per family:

- Column transformation: selective 17, tools 18.
- Deduplication: selective 13, tools 14.
- Filtering: both 14, plus the one shared incorrect acceptance.
- Missing values: selective 17, tools 16.
- Type conversion: selective 17, tools 16.

The tool parser makes eight tool attempts across four tasks: four successful `profile_column` queries and four `__unknown_column__` failures. This is the previously identified training recovery-prefix artifact. These parser queries are separate from the repair agent's tool calls. No operator normalization event occurs in either full run. The current results do not demonstrate a net benefit from parser tool access on this task set.

## Tokens and recorded time

Selective facts uses **513,385 total tokens**, including 177,514 parser tokens and 335,871 generation/repair tokens. Compared with 3B selective facts (537,681), this is 24,296 fewer, or 4.5% less. The repair agent makes 99 model calls and 48 tool calls, versus 104 and 49 for 3B.

Tools uses **688,063 total tokens**, including 258,796 parser tokens and 429,267 generation/repair tokens. Compared with 3B tools (632,678), this is 55,385 more, or 8.8% more. The repair agent makes 135 model calls and 71 tool calls, versus 110 and 46 for 3B.

The 7B tools method uses 174,678 more tokens than 7B selective facts: **34.0% more for equal total correctness**. Fewer tokens do not establish lower GPU cost across models with different sizes and precisions.

Summed recorded task time is about 37.5 minutes for selective facts and 39.4 minutes for tools, compared with 32.0 and 37.1 minutes in the 3B runs. These sums exclude setup/model loading and reference scoring; they are not notebook wall-clock duration or controlled throughput benchmarks.

## Recommendation and limits

Keep the unquantized 3B generator/repair version as the main controlled architecture experiment if the aim is to avoid giving the proposed system a stronger repair model than the direct baseline. Report 7B NF4 repair separately as a repair-capacity sensitivity experiment: it improves these results, but does not remove parser mistakes or guarantee better outcomes on every task.

Selective facts is the simpler, lower-token candidate among these two 7B runs. Preserve tools as a separate experimental method; these data do not support claiming it is superior.

For an architecture claim, complete the matching direct, gold-contract and other planned baseline runs with the same generator, task split, prompt version and resource accounting. The historical direct result of 78 correct/20 incorrect accepted/2 withheld is useful context, but comes from a different workflow and must not be presented as a clean paired baseline for this experiment. Do not use these repeatedly inspected 100 tasks as independent confirmation, or select configurations on them and then label them untouched test data.

No retraining is required to preserve or report these runs. Address contract semantic disambiguation, contradictory preservation rules, grouped-statistic routing and undeclared-column verification in separately versioned development experiments before another full evaluation.

## Reproduction and artifact locations

Run `.venv/bin/python archive/stage2_dual_7b/tools/analyze_stage2_dual_7b.py` from the repository root. It invokes the original strict collector from the sealed 7B source ZIP, replays parser policy from saved replies, executes reference/initial/accepted-final code on CPU, and saves detailed mismatches and task-level verdicts. The completed replay covers all 200 method/task records and finds **zero parser-policy mismatches, zero score mismatches and zero generation-input mismatches**. The original frozen-source integrity check also passes. No model inference is performed locally. CPU replay uses local Python/pandas; tied top-value display ordering is excluded from observation comparison, while statistical values and selected tool actions are checked.

- `audit.json`: detailed checks and task outcomes.
- `repair_model_comparison.json`: checked paired 3B/7B comparison.
- `comparison/summary.json` and `comparison/SUMMARY.md`: original strict collector output.
- Output ZIPs: `archive/stage2_dual_7b/kaggle_outputs/dual_parser_v1/`.
- Executed notebooks: `archive/stage2_dual_7b/notebooks/executed/dual_parser_v1/`.

Original upload bytes are preserved and recorded in `docs/uploaded_artifacts.json`. Runtime code, prompts, adapters and experiment sources were not modified.
