# Combined 3B repair: completed end-to-end results

**Both parser methods finish with 74 correct accepted outputs, one incorrect accepted output and 25 withheld tasks.** Compared with their corresponding original 3B repair runs (73 correct, two incorrect accepted, 25 withheld), this is one additional correct completion and one fewer incorrect acceptance. It is a modest improvement, with higher total token use. The 17-case repair-only improvement does not transfer at the same scale to the full workflow.

## Completion and comparison validity

Both uploaded full bundles cover 100/100 test tasks, have valid CRCs and SHA-256 seals, and pass the unmodified strict collector from their exact sealed source package. Both executed notebooks have no recorded exception outputs. Runtime source, input hashes, task inventories, selected adapters, model revisions, parser settings and software versions match the intended `stage2-combined-3b-fp16-v1` experiment.

Both generator and repair agent are Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, with actual floating parameters and computation recorded as torch.float16 and quantization=none. The selected method-specific parser adapters remain over Qwen2.5-1.5B-Instruct NF4. They are not new 3B parser adapters.

Compared with the original 3B run for each method, **parser payloads match after excluding elapsed time/backend path labels; generation payloads match after excluding elapsed time**. The contracts and initial programs are therefore the same. Model configuration, adapters, data, generator prompt, limits and software environment match. This is a controlled comparison of the combined repair bundle (evidence, compact history, recovery, operator conventions and schema guard), not a test of a changed parser or generator.

Source manifest: `d9290a6ccb28b15d544a306a6d88f77a91bc84e455078a19e0b24e7bd4004ac2`. Environment: Python 3.13.16, Torch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0, pandas 2.3.3 and NumPy 2.1.3, on T4 x2. Sequential model residency and the same three-proposal, eight-tools-per-attempt and 60,000-token limits are retained.

## Overall outcomes and where the gain comes from

For both selective facts and trained tools, final outcomes are 75 ACCEPT (74 correct, one incorrect), 12 ESCALATE and 13 REJECT.

Both gain the same two tasks relative to the original 3B loop:

- `adult_dedup_003`: after a failed merge/helper-column approach, the agent switches to dropping duplicates on fnlwgt while preserving the original columns. A fresh recovery is requested before success.
- `taxi_col_004`: the agent changes scalar Python rounding to NumPy rounding with `decimals=1`, satisfying the declared operator convention. No fresh-recovery request is needed for this success.

Both lose `airbnb_dedup_005`, which the original 3B agent repaired. The combined agent unnecessarily removes rows with missing last_review before deduplication. The request only asks for the first listing per host/neighbourhood; deleting those input rows causes 36 groups to lose their required first record. It repeats that approach until escalation. The stronger 7B repair experiment also failed this case. More elaborate feedback is not uniformly better.

Thus **two gains minus one loss = one additional correct completion** per method.

`taxi_filt_004` changes from incorrect ACCEPT to REJECT. The guard prevents acceptance of an undeclared helper column. The agent then overwrites the original timestamp strings with datetime values, violating preservation, and fails to return a correct accepted result. This is improved rejection behavior, not an additional successful repair.

## Initial correctness and withholding

Selective facts starts with 76 correct generated outputs, loses five to the contract/acceptance process, and repairs three initially incorrect outputs: **76 − 5 + 3 = 74**.

Tools starts with 77 correct generated outputs, loses nine, and repairs six: **77 − 9 + 6 = 74**.

The numbers of initially correct outputs lost are unchanged from the old loop. Repair-only changes have not solved this contract bottleneck.

The 25 withheld selective-facts tasks consist of five parse stops, 11 whose contracts do not accept the correct reference, and nine with reference-compatible contracts but no successful accepted repair. Tools has one parse stop, 14 contract/reference mismatches and ten reference-compatible unsuccessful cases.

The corresponding reference-compatible withheld counts were also nine/ten before. Gains and regressions plus the newly withheld helper-column case offset each other. Reference compatibility is audit-only and is not supplied as live feedback; it does not prove contract completeness.

Selective facts still has 12 emitted contracts incompatible with the reference; tools has 15. The schema guard does not increase those reference-incompatibility counts: correctly declared reference transformations pass the guard whenever they passed the original verifier.

## The remaining shared incorrect acceptance

`airbnb_filt_002` still filters on neighbourhood instead of the borough-valued neighbourhood_group. It returns zero rows instead of the reference's 927, satisfies the wrong predicted contract and is accepted by both methods.

The schema guard checks permitted column sets; it cannot detect that a valid input column was assigned the wrong semantic role. This failure needs request-to-schema/category-value grounding before contract admission, or an evidence-led parser correction. No such parser change was included in this integration.

## Remaining repair failures and transfer from the small suite

The small synthetic repair-only suite gave combined 16/17 versus control 11/17. These end-to-end runs instead give 74/100 versus 73/100. Those numbers are different experiments and denominators. The synthetic fixtures supplied reference-compatible contracts, short requests and small schemas; the real workflow includes malformed/incorrect contracts, larger data and different schema/row evidence. Success on a failure mechanism did not establish broad transfer.

Concrete remaining examples:

- `airbnb_conv_005`: the model continues calling set_categories without assigning its returned value, and adds categories already present, causing additional failures. Both related synthetic fixtures were solved, but this real task is not.
- `titanic_col_003`: it keeps Python scalar round at two decimals, including apply(round,args=(2,)), rather than using the declared NumPy convention. This echoes the unsolved two-decimal synthetic fixture.
- `titanic_col_001`: it repeatedly removes SibSp/Parch while adding num_relatives and eventually refers to those columns after dropping them. Better schema evidence alone did not make it preserve the inputs.
- Input-order deduplication and retaining unknown-age rows still have unsuccessful real cases despite synthetic successes.

Selective facts requests fresh recovery on 19 tasks; tools on 20. Only `adult_dedup_003` ends correctly among those recovery-triggered tasks in each method. Nine recovery-triggered tasks per method already have reference-incompatible contracts. This is an association within the traces, not a standalone causal estimate of the recovery feature; the entire repair prompt and evidence presentation changed together.

The remaining facts-router issues (ordinary “mean” interpreted as a statistic; grouped Age medians unsupported) and tool parser errors (querying Pclass instead of Age grouped by Pclass, incorrect preservation/null-count conditions) are unchanged. Tools still makes eight parser tool attempts across four tasks, with four placeholder failures.

## Selective facts versus tools

They have equal totals but different correct task sets. Tools succeeds where selective facts fails on `airbnb_dedup_003`, `taxi_col_005`, `taxi_miss_005`. Selective facts succeeds where tools fails on `taxi_conv_002`, `taxi_miss_003`, `taxi_miss_004`. This is the same three-for-three parser trade-off as before combined integration.

Tools has no aggregate correctness advantage and consumes **23.2% more total model tokens** than selective facts. On these runs, selective facts remains the simpler lower-token option; the results do not establish that trained parser tool access is superior.

## Cost and timing

- Selective facts: **556,724 total tokens** versus 537,681 originally, an increase of 19,043 (**3.5%**). Parser cost stays exactly 177,514; generation/repair cost rises from 360,167 to 379,210. Repair model calls fall from 104 to 82 and explicit repair tool calls from 49 to 23.
- Tools: **685,765 total tokens** versus 632,678 originally, an increase of 53,087 (**8.4%**). Parser cost stays exactly 258,796; generation/repair cost rises from 373,882 to 426,969. Repair model calls fall from 110 to 97 and explicit repair tool calls from 46 to 25.

Fewer calls do not imply fewer tokens: the revised prompts/evidence are larger on real tasks. Automatic evidence computation is additional work even when explicit tool calls fall. Recorded diagnostic construction totals about 8.5 seconds for selective facts and 9.2 seconds for tools.

Summed task time is about 33.2 minutes for selective facts and 36.2 minutes for tools, versus 32.0 and 37.1 minutes originally. These exclude setup/model loading and reference scoring, and are not controlled notebook-duration or throughput benchmarks. The synthetic notebook's 26.1% token reduction should not be claimed as a general end-to-end efficiency result.

## Development split shown in the notebooks

The executed notebooks also print completion and summaries for the 20 end-to-end development tasks:

- Selective facts: 14 correct, zero incorrect accepted, six withheld; no initially correct output lost and zero successful repairs.
- Tools: 13 correct, one incorrect accepted, six withheld; one initially correct output lost and zero successful repairs.

These figures come from notebook output. The supplied ZIPs are the test bundles; the separate development ZIPs were not provided here, so their per-case outcomes have not been independently replayed in this review. The printed development result already cautions against treating the synthetic repair success as a reliable full-system fix.

## Recommendation

Retain combined as a documented development variant: it improves this pair of runs slightly and prevents one known class of false acceptance. Do not describe it as a major repair breakthrough or a cost reduction. Report the regression, token increase, and residual wrong acceptance alongside the gain.

The next useful work is **targeted contract/data grounding and contradiction checks on development tasks**, rather than another wider rerun or increasing repair attempts. Prioritize exact column/key resolution, missing-value/count contradictions, category/order semantics and grouped statistics. Correct code cannot satisfy an incorrect contract reliably. Preserve the current model sizes and evaluate parser changes separately from further repair prompting.

For repair, a narrow general operator/API aid for pandas assignment, input-order deduplication and explicit rounding arguments is a candidate for a separately versioned development test. The current results do not justify appending an unrestricted larger profile or more generic retry instructions. Selective facts is a reasonable first development target due to its lower token use; keep the tool method as a separate comparison.

If the research objective is final reporting now, present original 3B (73/2/25), combined 3B (74/1/25), and 7B NF4 repair (78/1/21) as separate configurations. The 7B comparison changes size and precision, and the historical direct baseline belongs to an older matched six-system experiment. Fresh matched baselines and independent tasks are still required for a stronger architecture claim.

## Reproduction and artifact locations

Run `.venv/bin/python tools/analyze_stage2_combined_3b.py`. It strictly collects from the matching sealed source, verifies saved parser policy/generator prompts, executes references and initial/accepted-final code on CPU, checks acceptance against both original and guarded verification, and saves detailed per-case evidence. The completed CPU replay of all 200 method/task records finds **zero parser-policy mismatches, zero score/verifier mismatches and zero generator-input mismatches**. Both notebook sources match the supplied templates. It does not invoke a model or silently replace existing scores.

- `audit.json`: CPU policy/score audit and per-task results.
- `paired_control.json`: checked paired comparison with original 3B runs.
- `comparison/summary.json` and `comparison/SUMMARY.md`: genuine strict collector results.
- Uploaded ZIPs: `kaggle_outputs/stage2/combined_3b_v1/`.
- Executed notebooks: `notebooks/executed/stage2/combined_3b_v1/`.

Original upload bytes are preserved in the upload index. No experiment source, prompt, adapter or result was modified during analysis.
