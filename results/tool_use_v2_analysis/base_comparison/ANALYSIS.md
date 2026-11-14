# Direct base-model tools: 1.5B and 3B

Both notebooks completed their full 100-case parser-development runs without notebook exception outputs. The result ZIPs are complete, non-smoke evaluations. Neither loaded a contract adapter: `adapter_hashes` is empty, with the expected pinned base IDs/revisions and `parser-tools-v2`.

## The notebook filenames were reversed

- Uploaded `stage2-tools-nf-3b.ipynb` actually sets `MODEL_SIZE = "1_5b"` and ran the 1.5B model.
- Uploaded `stage2-tools-nf-15b.ipynb` actually sets `MODEL_SIZE = "3b"` and ran the 3B model.

The result ZIPs are correctly labeled. Interpretation follows the code and result metadata, not the uploaded notebook names. The executed notebooks are now stored with their actual model sizes under `notebooks/executed/tool_use_v2/`; the ZIPs are under `kaggle_outputs/tool_use_v2/`. Training must likewise be identified from `MODEL_SIZE`, `run_info.json` and adapter metadata, not the notebook filename.

## Overall results

**1.5B:** 14/100 correct-reference passes; 22/100 tasks accept at least one tested wrong output; only **10/100** both accept correct outputs and reject all available wrong probes. It returns 49 schema-valid contracts. The other 51 outcomes comprise 43 invalid-response stops, six token-budget stops, one tool-limit stop and one explicit escalation.

**3B:** 6/100 correct-reference passes; 13/100 tasks accept at least one tested wrong output; only **4/100** satisfy both checks. It returns 40 schema-valid contracts. The other 60 cases each produce an invalid first reply and then explicitly escalate after feedback.

Correct-reference and wrong-output acceptance are separate tests and can overlap: four 1.5B cases and two 3B cases pass a correct output while also accepting a tested wrong one. Therefore 14 and 6 overstate the number of useful contracts under these probes. Conversely, fewer wrong acceptances from a model that frequently escalates do not establish better safety.

These are **intent-parser development scores**, not generated-code accuracy, Stage 2 outcomes, or results for the fine-tuned models currently training. The 80 synthetic cases use targeted wrong outputs; the 20 original development tasks use only wrong no-op probes. Neither set establishes general safety or independent research accuracy.

## Actual tool behavior

**1.5B issues 56 tool attempts on 16 tasks.** Forty-one calls return successful tool observations and 15 fail. It calls `profile_column` 26 times, `inspect_rows` 10 times, `group_profile` nine times and `count_rows` twice. It also invents unavailable modifying tools: `rename_column` seven times and `fill_missing` twice. Other errors include unknown columns, overly broad group queries and an unsupported argument.

None of the 16 tasks with any tool attempt passes its correct-reference check. The tool functions can return useful facts, but the model does not reliably turn those observations into a valid, useful final contract. It also spends tools on tasks such as renaming and dropping missing rows where it does not need a computed median or mode.

**3B issues zero tool calls across all 100 tasks.** This is zero attempts, not successful tool use or proof of tool reliability. All calls available to either model are read-only; modifying the input is not a valid parser tool action.

Among the 16 synthetic statistic-dependent tasks, **both models pass 0/16**. The 1.5B model attempts tools on only three of them: `solar_fill_0`, `solar_fill_1` and `warehouse_fill_2`. The first two obtain useful observations but loop or fail to finish. The third repeatedly invents `fill_missing`, which is not an available tool. No grouped-median task is solved by either model.

## Concrete failures

- **1.5B, `solar_fill_0`:** obtains the full input profile containing the median, then repeatedly alternates profile and sample queries. Six successful tool observations still do not yield a valid final contract.
- **1.5B, `warehouse_fill_2`:** asks for `fill_missing` twice. The parser's tools inspect input data; they do not perform transformations. The runtime correctly returns an unknown-tool error.
- **1.5B, `adult_col_006`:** changes `education-num` to the nonexistent `education_num`, uses malformed preservation entries, then requests nonexistent `rename_column`.
- **1.5B, `warehouse_fill_0`:** returns a schema-valid contract demanding that the target values stay unchanged, contradicting a fill. Schema validity alone is insufficient.
- **3B, `warehouse_fill_0`:** demands unchanged values/columns and accepts the wrong unchanged input while rejecting the correct median fill.
- **3B, `warehouse_fill_3`:** emits unsupported `fn="median"` instead of querying a statistic and using `fn="const"` with its value; after schema feedback, it escalates.
- **3B, `warehouse_dedup_0`:** uses `must_preserve: ["*"]`, which is not a list of predicate objects; it escalates after the error instead of correcting the format.

The 1.5B model's correct synthetic references are mostly deduplication and some type conversions; it solves no synthetic filtering, fill or general edit case. The 3B model solves only two synthetic clipping cases. These results show weakness across the contract vocabulary, not just in fetching missing statistics.

## Cost

The 1.5B run records **617,444 tokens across 245 model replies**, with summed parser runtime approximately **21.7 minutes**. The 3B run records **354,294 tokens across 160 replies**, approximately **15.5 minutes**. These durations exclude setup, model loading, CPU reference/probe execution and the separate smoke run. Exact per-call tokenizer counts are not saved; totals are the recorded parser counters, checked against their aggregate summaries.

The 3B run is cheaper partly because it commonly escalates after two replies. The 1.5B model's repetitive tool and correction loops add cost without producing useful contracts on tool-using tasks. These timings do not establish a general speed ranking for the model sizes.

## What this means for the training runs

Let the independently configured **1.5B and 3B tool-training jobs finish**. Their matching multi-turn supervision is intended to teach exactly what the base models fail here: valid predicate objects, distinguishing read-only inspection from transformation, requesting missing input facts, using observations, and finishing with the contract. Each training job runs its own same-size base evaluation, so neither depends on the separate ZIPs analyzed here.

Training is a plausible next step, not a guarantee. Check the actual size in each job's configuration because the two uploaded base notebook filenames were swapped. There is no need to change a correctly configured job or modify its running source based only on these baseline scores.

When the training outputs arrive, assess at least:

1. Correct-output acceptance **and** rejection of tested wrong outputs, including per-pattern regressions.
2. Statistic-dependent task success, queries to the correct column and scope, and correct use of returned values.
3. Whether ordinary rename/deduplication/literal-fill requests remain efficient without unnecessary queries.
4. Fewer malformed contracts, invented tools, repeated queries and budget stops.
5. Model replies/tokens and semantic results at the same model size and protocol.

Do not select solely on number of tool calls or the training `eligible` flag. The declared rule compares against these weak same-size bases; even a modest absolute result may satisfy it. As additional context, the existing original contract adapter passed 68/100 development references and selective-facts epoch 2 passed 73/100, though those use different input protocols and are not clean tool-use ablations. A tool adapter should demonstrate practical quality before replacing either pipeline.

This experiment uses a **custom JSON text protocol plus a detailed contract vocabulary**, not a model's native function-calling interface. The result does not show that all untrained instruction models cannot use tools or that a larger model is generally worse. In this setup, 3B does not improve on 1.5B. Prompt structure, complete worked demonstrations, decoding and training are additional variables that were not isolated here. Preserve the current version while the matching training jobs run.

## Verification and reproduction

Run `.venv/bin/python tools/analyze_tool_use_base.py`. It checks ZIP CRC/seals, full task coverage, source/input hashes, base model/protocol metadata, notebook source except the model-size setting, summary arithmetic, exact input prompts, tool observations/control flow, and CPU semantic scores. The budget-stop location is taken from the saved trace because per-call tokenizer counts are absent. The audit makes no GPU calls and does not change either running training implementation.

Detailed verification, per-pattern results, examples and paired cases are saved in `audit.json`.

Audit completed on all 200 records: no initial-prompt, CPU semantic-score or counter mismatches. Both notebook sources match the launcher except their selected model size, and both have zero exception outputs. Three field-level trace differences affect two 1.5B tasks: a Python 3.13 versus 3.12 JSON error-message wording difference (`warehouse_filter_3`), and equal-frequency top-value ordering/cutoff membership (`titanic_conv_006`). The tied candidate values/counts were checked against the input; statistics and modes match. Neither case passes its correct reference, and terminal statuses/actions/counters reproduce. These replay differences are retained in `audit.json`; exact token-budget retokenization was not performed.
