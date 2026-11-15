# Completed 3B repair-development comparison

The combined evidence/recovery and schema-guard version gives **16 correct accepted outputs, zero incorrect acceptances and one withheld case out of 17**. The original a1-v4 control gives **11 correct, one incorrect accepted and five withheld**. The combined arm gains five correct cases without losing a control success and uses 26.1% fewer recorded model tokens.

This is encouraging evidence for integrating the combined repair version into a separately versioned Stage 2 experiment. It is a small synthetic repair-development result, not a measured increase from 73/100 in the real Stage 2 workflow.

## Completion, configuration and evidence

The executed notebook matches the supplied template and has no recorded exception outputs. The bundle is complete with 68/68 expected task/arm records, four arms on the same 17 fixtures, and `model_quality_result=true`. It is the full run, not the CPU scripted validation or two-case GPU smoke. ZIP CRC and SHA-256 file seals pass. The source manifest is `4be9feb5542ba53281961a772f97154d85836caab92af66f931d647d195521aa`; local runtime hashes, suite hash, task inventory and declared limits match.

The resolved model is **Qwen/Qwen2.5-3B-Instruct**, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, on CUDA 0. Recorded floating weights and compute are **torch.float16**, with **quantization=none**. Hardware is Tesla T4, with Python 3.13.16, Torch 2.11.0+cu130, Transformers 5.20.2, Accelerate 1.15.0, pandas 2.3.3 and NumPy 2.1.3.

All arms use three repair proposals, eight tool calls per attempt, a 60,000-token budget and a 768-token output cap. There is no parser, adapter training or initial code-generation call in this experiment. Starting programs and fixture contracts are supplied by the development suite. The same references are used only to score results; contract-derived expectations in diagnostics are not reference output.

## Four-arm results

- **Control:** 11 correct, one incorrect accepted, five withheld; six successful repairs; 39,194 tokens, 30 model calls and six explicit tool calls.
- **Evidence/recovery:** 15 correct, one incorrect accepted, one withheld; ten successful repairs; 27,434 tokens, 17 model calls and zero explicit tool calls.
- **Schema guard:** 12 correct, zero incorrect accepted, five withheld; seven successful repairs; 40,354 tokens, 31 model calls and six explicit tool calls.
- **Combined:** 16 correct, zero incorrect accepted, one withheld; eleven successful repairs; 28,976 tokens, 18 model calls and zero explicit tool calls.

All five already-correct controls are accepted without a model call in every arm. No initially correct output is lost. Those controls test the acceptance/guard path; they do not show that the model avoids regressions while repairing code blocked by an incorrect contract.

The combined version goes from six to eleven successful repairs among the 12 flawed starting programs. Its final 16/17 correctness is 94.1%, compared with control's 64.7%, but the denominator is too small and targeted for a general accuracy claim.

## Exactly which tasks improve

Evidence/recovery gains four tasks relative to control, with no losses:

1. `category_assignment_1`: assigns the returned `set_categories` result back to the column, obtaining the requested ordered categories.
2. `category_assignment_2`: assigns the returned ordered `reorder_categories` result.
3. `rounding_convention_1`: uses `np.round` with `decimals=1`, matching the contract's NumPy rounding convention.
4. `filter_keep_unknown`: retains rows with missing measurements through an explicit `isna()` alternative alongside the numeric threshold.

The schema guard fixes `filter_extra_helper`. Control and evidence-only accept the starting program with an undeclared `temporary_stamp` helper column. Guard-only and combined identify the unexpected column, invoke repair, and correctly return only `record` and `stamp`. This is an actual correct repair, not merely conversion of incorrect ACCEPT to withholding.

Combined includes all five gains. Its remaining successes match control. The complete-record deduplication case is also more efficient: control and guard-only need three proposals, whereas evidence and combined succeed with one. Several other deduplication and preservation fixtures are already solved by control, so this run does not show an accuracy gain on every failure mechanism seen in the original 100-task analysis.

## What the recovery mechanism demonstrates

Evidence and combined each request a fresh recovery on four tasks: both category cases, preservation of unknown measurements, and the two-decimal rounding case. The first three eventually succeed; two-decimal rounding remains unsuccessful. The one-decimal rounding improvement occurs without a fresh-recovery request.

Therefore recovery is useful in these traces but is not independently proven to cause every gain. The evidence arm changes examples, history presentation, operator conventions, output provenance and recovery together. An ablation would be needed to identify their separate contributions.

Zero explicit repair tool calls in these arms means the model did not need to request a tool. Automatic diagnostic computation and candidate execution still occur. The run records diagnostic construction time, and all submitted candidates still pass through execution and verification.

## The one remaining failure

`rounding_convention_2` fails in every arm. The request explicitly asks for pandas/NumPy rounding to two decimals, so the expected convention is stated in this fixture.

For combined, the trace shows:

1. The starting scalar Python `round(x, 2)` produces 21.07 instead of 21.08, 31.27 instead of 31.28 and -9.47 instead of -9.48.
2. The first proposal adds null handling but keeps scalar Python rounding; the same three rows remain wrong.
3. A fresh-recovery request is issued. The model switches to `apply(np.round)` but omits the decimal argument, rounding to integers instead.
4. Its final proposal returns to the starting scalar-round program. Repeated-code detection catches it, and the three-proposal cap ends the run as REJECT.

The evaluator/guard is doing its job by withholding the wrong program. The model has not consistently translated the explicit rounding convention into the correct API call. A vectorized `Series.round(2)` is the fixture reference implementation, not something supplied as the answer during this live run.

There is no need to retrain to pursue this remaining issue. A general operator help card that includes argument semantics or a bounded API lookup is a possible later development experiment. Do not hard-code this task's output values or alter the existing run's score. One unsolved fixture is not a reason to expand the attempt budget for all tasks or delay validating the broader gains.

## Tokens and timing

Compared with control, evidence-only uses **11,760 fewer tokens (30.0% less)**. Combined uses **10,218 fewer tokens (26.1% less)** and 18 instead of 30 model calls. Guard-only adds 1,160 tokens because it correctly opens a repair episode for the otherwise falsely accepted helper-column case.

Summed recorded repair time is 85.1 seconds for control, 58.5 for evidence, 86.4 for guard-only and 63.1 for combined. Combined is about 25.8% lower on this run. These are per-repair sums, excluding dependency installation, model loading, attention preflight, fixture validation, CPU checks, smoke and notebook orchestration. Treat them as recorded observations, not a controlled hardware-throughput benchmark.

Automatic evidence construction totals about 0.048 seconds per evidence-enabled arm across these small fixtures. This does not predict diagnostic overhead on larger real datasets.

## What to do next

Use **combined** as the candidate for the next Stage 2 development version, with generation and repair still at unquantized 3B FP16. Integrate it into both selective-facts and trained-tool parser workflows under a new experiment identifier. Initially keep the parsers and their input protocols unchanged so the end-to-end effect of the repair change can be measured. Keep the existing baseline/control runs and source seals intact.

First run an end-to-end development check with correct and incorrect contracts, realistic dataset sizes and representative successful tasks. This notebook uses fixed compatible fixture contracts, so it cannot establish that the five/nine initially correct outputs blocked by bad contracts in the original Stage 2 runs will be recovered. The wrong-column Airbnb contract, grouped-statistic routing and null-count contradictions remain outside this notebook's fixes.

Once that integration passes, rerun the affected proposed-system comparison before deciding whether to invest in all matching baselines. Any claim about the architecture still needs the appropriate matched baseline configuration and clear cost accounting. Use fresh held-out tasks for independent confirmation; the original 100 cases and this targeted suite are already development-informed.

## Audit and artifact locations

Run `.venv/bin/python tools/analyze_stage2_repair_dev.py` to validate seals, source, suite, metadata, notebook and aggregates, then replay all 68 saved response sequences on CPU. The audit checks outcomes, accepted-output equivalence, final code, full event sequences excluding timings/token fields, and recorded per-task token/model/tool counts. Local scripted replay token estimates are not substituted for actual recorded model usage. See `audit.json` for the verified checks and per-task comparison.

- Canonical output: `kaggle_outputs/stage2/repair_dev_v1/stage2_repair_dev_full.zip`.
- Executed notebook: `notebooks/executed/stage2/repair_dev_v1/stage2_repair_dev_kaggle_executed.ipynb`, originally `repair-dev.ipynb`.
- Current report: `results/stage2_repair_dev_analysis/v1/ANALYSIS.md`.

The original uploaded bytes are preserved and indexed in `docs/uploaded_artifacts.json`. This review adds analysis only; no frozen runtime or score was modified.
