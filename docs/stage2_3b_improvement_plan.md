# Plan for improving the 3B Stage 2 system without retraining

Status: failure review completed. The first repair-evidence/recovery stage and independently switchable schema guard are implemented in `eval/stage2_repair_dev/`, with a paired development notebook. See [run instructions](stage2_repair_dev.md). Contract correction and information routing below remain planned; the completed synthetic repair-only GPU comparison is available in [the verified analysis](../results/stage2_repair_dev_analysis/v1/ANALYSIS.md): control 11/1/5 versus combined 16/0/1 (correct/incorrect accepted/withheld). Both end-to-end integrations are now implemented; see [the combined workflow](stage2_combined_3b.md). Their completed results are 74 correct, one incorrect accepted and 25 withheld for each method; see [the end-to-end analysis](../results/stage2_combined_3b_analysis/ANALYSIS.md). The gain is modest and token use increases. Keep generator and repair Qwen2.5-3B-Instruct in FP16, with the existing selected 1.5B parser adapters. Preserve both selective-facts and tool-parser methods.

## Evidence and what the current numbers mean

Reviewed both complete 3B bundles, the existing CPU-verified per-task audit, all unsuccessful contracts and the repair traces for reference-compatible failures. There are 54 unsuccessful method/task outcomes across 30 distinct tasks: each method has 25 withheld outputs and two incorrect acceptances. Full inventory: `results/stage2_dual_3b_analysis/failure_review/INVENTORY.md` and `inventory.json`.

Selective facts: five parsing stops, 11 withheld contract/reference mismatches, nine withheld cases with reference-compatible contracts, two incorrect acceptances. Five initially correct outputs are withheld.

Tools: one parsing stop, 14 withheld contract/reference mismatches, ten withheld cases with reference-compatible contracts, two incorrect acceptances. Nine initially correct outputs are withheld.

Therefore repair-only changes principally target the nine/ten reference-compatible withheld cases. This is a diagnostic opportunity count, not a promise of that many gains: a reference-compatible contract can still be incomplete, and changing repair can also regress successful cases. Other failures require parser, information-routing or verifier changes.

Both 7B comparisons gain six correct cases and lose one, but preserve the five/nine initially correct outputs lost. This supports improving repair capability while independently addressing contract problems.

## What already exists — do not duplicate it

`src/igt/agents/repair_agent.py` already passes full expected/observed reports and input row indices, keeps attempt history, exposes tools, hashes normalized code, warns about exact repeated proposals and stops after two repeats. Its prompt already says to preserve unrelated columns/types and remove temporary columns.

`src/igt/pipeline/failure_report.py` produces meaningful reports, not merely predicate IDs. Examples replayed from these runs:

- Ordered category: ordered=False when ordered=True is required.
- Taxi deduplication: 99 of 692 key groups do not keep their first original row.
- Age filter: 177 of 778 rows that should be retained were removed.
- Rounding: expected value and actual value at an example input index are already reported.

`src/igt/agents/repair_agent_v5.py` contains an existing opt-in discipline prompt. It changes the prompt but not the a1-v4 loop. It is a useful optional development comparator; appending more generic instructions is not itself a new architecture.

`finetune/selective_facts_correction/checks.py` already detects some unknown columns and change/preservation conflicts. The deployed dual workflows use normalization only. Earlier generic one-correction development runs produced no additional correctness gain, so reuse and extend concrete diagnostics rather than assume another generic retry will help.

## Priority 1: make existing evidence actionable for 3B repair

Create a separate repair version with a compact, automatically constructed diagnostic message for the latest candidate:

1. Include exact current code, failure stage, failed requirement, relevant input schema and whether the candidate actually produced an output.
2. Materialize at most three to five relevant counterexamples. For a value relation, show original value, the expected value derived from the predicted contract, and observed value. For preservation, show missing/extra columns or changed cells. For deduplication, show the input-order group and the actually retained row. For categorical types, show observed versus required ordering and category list. Label these as contract-derived expectations, not independent truth.
3. Keep the original request adjacent to this short diagnosis. Summarize earlier attempts by failed checks and meaningful code differences instead of repeatedly presenting an expanding history of near-identical programs.
4. After a repeated proposal or unchanged failure, switch once to a fresh-code prompt: ask for the simplest implementation of the request that satisfies the stated constraints, using exact schema names. Include the bad approach and its failure briefly. Keep the three-proposal and 60,000-token limits, count the reset as part of the same budget, and never accept an unchecked candidate.
5. Track output availability and the code hash that produced each tool observation. A crash should explicitly say no output exists for that candidate; guide inspection toward the input. Retain useful observations across attempts with their provenance. Do not silently label an older output as the current result.

Concrete targets:

- `airbnb_conv_005`: the return value of set_categories is ignored; assignment is required. Generic repeated-code warnings already failed.
- `adult_dedup_003`: replace a broken merge/helper-column approach with a straightforward operation that keeps actual input records; avoid inspecting last_output after a crash.
- `titanic_col_001`: adding num_relatives must not remove SibSp/Parch or other input columns.
- `taxi_dedup_005`, `titanic_dedup_002`: first/last refers to original input order. Sorting before deduplication can select a different row.
- Tools `airbnb_dedup_004`: filtering to size-one groups removes repeated groups entirely, so keeping last afterward cannot recover them.
- `titanic_dedup_004`: groupby.first can combine first nonmissing cells into a row rather than preserve a complete original record.
- `titanic_filt_003`: unknown ages must be retained; do not fill or discard them.

These should become general development examples with different data/column names, not task-ID-specific repair rules.

### Rounding needs explicit specification

CPU reproduction confirms that Python scalar round and the current NumPy-based contract differ on 42 taxi rows and 29 Titanic rows. Examples: round(1.05,1) gives 1.1 while the current vector convention gives 1.0; round(21.075,2) gives 21.07 while the vector convention gives 21.08.

Treat this as an interface/specification issue as well as a repair problem. Document the contract's NumPy/pandas rounding convention and expose a compact general operator-semantics description to the generator and repair agent in a new version. Test ties, negatives, nulls and ordinary values. Do not loosen tolerance by a decimal unit or relabel previous results. For human requests where a different rounding policy is legitimate, specify or clarify that policy; do not claim the ordinary phrase "round" uniquely implies the benchmark's convention.

## Priority 2: repair contracts only when there is identifiable evidence

Add a bounded contract-diagnostic stage before code repair. Keep contract editing outside the repair agent: once a contract is admitted for a code-repair episode, the repair model cannot weaken it to get a PASS.

Extend input-only checks for:

- Unknown input columns, invalid operators/JSON and preservation of a newly added column.
- Required edits that conflict with preserving those cells, or filling nulls while requiring the null count to remain unchanged.
- Null-count changes inconsistent with explicitly specified replacements and preserved remaining cells, where the conflict can be proven.
- Numeric comparisons applied to textual categorical columns, accounting for genuinely numeric/categorical representations.
- Mapping keys incompatible with actual input values; distinguish a literal key "Other" from a default branch. Missing mapping coverage is evidence to inspect against the request, not automatic proof of error.
- Ordered categories and value remapping as distinct operations.

On a proven schema/inconsistency error, allow at most one parser correction with a concise, structured diagnostic and the original request. Revalidate the complete result. If it still fails, withhold with the exact reason. Keep the initial generated candidate available and recheck it after a legitimate corrected contract, rather than unnecessarily modifying already-correct code.

For semantic ambiguities, gather evidence first. Example: find which candidate text columns actually contain request values such as borough names; show bounded matches and exact names to the parser. The presence or absence of a value alone is not proof of intended semantics, and an empty result can be legitimate. Do not automatically choose another column or reject every empty output.

This targets the lost initial successes and the Airbnb false acceptance. Important cases include wrong deduplication keys, invented listing_name, incompatible null preservation, wrong boolean-map keys, a threshold encoded as a literal map key, and extra Cabin/Fare conditions invented for "travelling alone".

If evidence-led correction still fails in development, report that limitation rather than add unbounded retries or silently discard inconvenient checks. No retraining is required for this experiment.

## Priority 3: correct information routing

Selective facts:

- Distinguish statistical "mean of column X" from ordinary "these values mean not recorded"; test both positive and negative language cases.
- Support explicit group-by statistics, retaining group-key types, full precision, missing group keys and all-missing groups. Do not silently substitute a global median when a group's statistic is unavailable.
- Supply input-derived category/order pairs for requests such as education ordered by education-num; avoid giant profiles on unrelated tasks.

Tools:

- Add a concise precondition that named columns exist before issuing a data query; return usable exact-name guidance for placeholder requests.
- Make the distinction between a value column and grouping columns explicit in the tool instructions. A group median needs Age as the value column and Pclass as the key, not a median of Pclass.
- Test scoped examples on development tasks. Do not force every request through a statistics query or present automatic query selection as model-selected tool use.

Maintain the distinction between the two methods: automatic facts for selective facts, model-selected queries for tools. A deterministic routing fallback would be a separately labeled hybrid variant.

## Priority 4: prevent undeclared output-schema changes

Add a versioned exact-schema guard to catch the helper-column false acceptance in `taxi_filt_004`. Compute permitted additions/removals/renames from explicit contract declarations; preserve legitimate additions and renames and reject unaccounted changes. Return a precise schema difference to the repair loop.

A wrong contract can still authorize a wrong change, so this guard complements semantic contract checks. It must not forbid all new columns, auto-delete columns, or replace full intent verification. Test unchanged schema, legitimate add/drop/rename operations and unexpected helper columns. The guard can initially increase withholding; count a case as fixed only if the final accepted output is correct.

This small safety fix can be developed alongside Priority 1; evaluate its contribution separately so changes in false acceptance are not attributed to the repair prompt.

## Development experiment and delivery sequence

1. Preserve the frozen 3B workflow and existing results. Record a new protocol/version for every changed prompt, routing rule or verifier policy.
2. Build development cases representing the failure mechanisms above using different values, schemas and task wording. Include successful renames, fills and ordinary filtering as regression controls, legitimate empty outputs, valid added columns and ambiguity cases. Current 100 test cases remain a diagnostic regression set because they have already informed this plan.
3. CPU checks first: validate diagnostic truthfulness, original-row alignment with duplicate indices, guard behavior, schema scope, rounding conventions and unsupported/ambiguous cases. Audit reference solutions and targeted wrong programs; reference code and gold results never enter live prompts or routing.
4. One focused development notebook with selectable arms: current 3B control; repair evidence/reset only; parser/routing diagnostics only; combined version with the schema guard. Include a guard-only CPU audit and optional existing a1-v5 prompt comparator. Run each arm independently with fresh contract/generation output when its prompts change. A repair-only microbenchmark may explicitly reuse the same failed starting candidate to isolate repair, but label it separately from end-to-end results.
5. Keep model precision and proposal/token limits fixed. Report every extra parser call, automatic diagnostic operation, tool execution, elapsed time and token cost. If a new operator-semantics card changes the generator prompt, test it as a distinct arm rather than attributing its benefit to repair alone.
6. Select on development: improve correct accepted outputs, prevent increases in incorrect acceptances, reduce initially correct outputs lost, and quantify cost. Examine per-task gains and regressions, including previously successful cases. A variant that merely converts wrong ACCEPT to withheld is safer but has not improved task completion.
7. Freeze the selected version, then generate the matching Stage 2 notebooks for both methods and the comparable baselines. Use a fresh held-out set for a confirmatory research claim; report old-100 results as diagnostic follow-up.

The first implementation deliverable should be the compact diagnostic/reset repair version plus the isolated schema guard, a small development suite and one comparison notebook. Extend contract diagnostics and routing as the next separately measured change. Do not start by rerunning all six full test notebooks or increasing model size/attempt budgets.

## Expected outcome and boundaries

The plan directly addresses observed mechanisms, but no exact score increase is guaranteed. Repair evidence may help the nine/ten compatible-contract failures; contract work may recover some of the five/nine already-correct candidates lost; the schema guard may prevent one known class of false acceptance. These groups overlap with other failure mechanisms and are not additive forecasts.

A strong architecture result would show that the same 3B model accepts more correct outputs while retaining low incorrect acceptance, with transparent compute costs and matching baselines. It should not rely on weakening verification or handing the system benchmark answers.
