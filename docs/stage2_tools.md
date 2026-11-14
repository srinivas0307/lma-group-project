# Separate Stage 2 parser-tool version

**First proposed GPU run completed:** 67 correct, 1 incorrect accepted, 32 withheld. The parser made **zero tool calls**; the repair agent separately made 67. This run does not demonstrate parser tool use. See the [validated tool-run analysis](../results/stage2_tools_analysis/proposed_v1/ANALYSIS.md). Keep the ongoing selective-facts training as a separate experiment; its adapter does not implement this tool protocol.

Experiment: **stage2-tools-v1**. Prompt/protocol: **s2-parser-tools-v1**. This is separate from `stage2-independent-v1`, which supplies an automatic data profile. You may finish the profile version and run this version afterward. No source or notebook in the current profile version was changed to enable tools.

## What changes

The parser initially sees the request, schema, missing counts and three sample rows. It does not receive the full computed profile. It decides whether to query the input dataset, receives an observation, and can ask again before returning a contract or escalating. A request that already specifies all needed values may legitimately require zero tool calls.

Available read-only tools:

- `profile_column(column, where?)`: numerical statistics at full available precision, missing counts, frequencies and modes on the requested subset. Mode ties and truncated value lists are explicit.
- `count_rows(where)`: number of rows satisfying the contract condition grammar.
- `inspect_rows(columns, where?, n?)`: at most five rows and eight specified columns, with full-precision values and explicit truncation.
- `group_profile(column, by, where?)`: profiles by one or two grouping columns, at most twelve groups; missing group keys are included.

Example conversation:

```text
User: Fill missing Age in Region west with the median Age of Region west.
Parser: {"tool":"profile_column","args":{"column":"Age","where":{"column":"Region","op":"==","value":"west"}}}
Program: executes that read-only query on the input CSV and returns its actual statistics.
Parser: writes a contract using the returned median, scoped to missing Age in Region west.
```

Tools cannot execute model-written Python, alter the CSV, read files/network, or inspect gold contracts, reference outputs or candidate transformations. Observations contain complete JSON; oversized results return an explicit error asking for a narrower query, rather than silently cutting off values. An unresolved mode tie requires a stated tie-break or escalation.

## Prompt and settings

The new system-prompt addendum explicitly permits tool-call JSON during information gathering and reserves contract formatting for the final answer. It explains arguments, conditional scopes, precision, failed queries and escalation. The interaction uses assistant tool-call text followed by a user observation message, compatible with the existing chat template; it is not native function-calling API machinery.

Limits per parser task:

- Six tool attempts, including failed queries.
- Ten model replies.
- Two corrections after malformed responses; a third malformed reply escalates.
- 24,000 cumulative parser input/output tokens. Another call is refused if its prompt plus the maximum output allowance cannot fit the remaining budget.
- 1,024 output tokens per reply; 12,288 context tokens for the 1.5B parser.
- 6,000 characters per observation; zero temperature and seed zero.

The prompted 7B parser shares the token/call/tool budgets; its backend has the existing 32,768-token context cap. Model-selected tools are also enabled for that comparison arm. The generator, verifier and repair behavior stays matched to the independent profile version; the repair agent's existing tools are separate from these new parser tools.

A parser that keeps requesting tools after the limit, cannot resolve missing information, exhausts its response budget, or produces invalid outputs ends with an explicit parser escalation. Infrastructure failures such as CUDA errors fail the job and do not masquerade as task outcomes.

**The original fine-tuned adapter was trained to emit contracts, not this interactive tool protocol.** This version is an exploratory inference experiment using those weights. A prompt change does not guarantee reliable tool use. Inspect tool uptake, failed calls, invalid contracts, false rejections, final outcomes and total costs before deciding whether to train a tool-aware adapter. No new training or GPU quality results are claimed here.

## Kaggle workflow

1. Download `notebooks/kaggle/stage2_tools/stage2_tools_notebooks.zip`.
2. Start with **`stage2_tools_05_proposed.ipynb`**. Attach your existing private 1.5B adapter dataset (`/kaggle/input/datasets/srinivasu2005/qwen-1-5b-adapter` is supported automatically), enable Internet and select **GPU T4 ×2**.
3. In a fresh Kaggle session use **Save Version → Save & Run All**. Setup includes the torchao compatibility fix and tiny LoRA probe. A two-task smoke run precedes the full 100-task run. Wait for **COMPLETE AND VALIDATED: 100 / 100**.
4. Download `stage2_tools_proposed.zip` and the executed notebook. Inspect `contracts/<task_id>.json`: it records every assistant reply, tool request, observation, tool failure, token cost and stopping reason.
5. For a complete comparison, run the other five `stage2_tools_01...06` notebooks independently. Only notebooks 05 and 06 require the 1.5B adapter. The 3B adapter is not needed. Each job generates its own outputs; no pilot or profile-version result is required.
6. Attach all six full **tool-version** ZIPs to `stage2_tools_collect.ipynb`, select accelerator **None**, enable Internet, and run all cells. Download `stage2_tools_summary.zip` after **600 fresh task records verified**. Do not mix profile-version and tool-version bundles in either collector.

Save full ZIPs under `kaggle_outputs/stage2/tools_v1/` and executed notebooks under `notebooks/executed/stage2/tools_v1/`. Keep the current profile-version files under their existing `independent_v1/` folders. Both reports can then be compared as separate experiments.

The raw original 100 tasks are embedded, so no GitHub push is required. They remain diagnostic because earlier failure analysis informed this work. A reviewed independent benchmark is still needed for fresh confirmatory evidence.

## Files and CPU checks

- `eval/stage2_tools/context.py`: parser prompt and initial messages.
- `eval/stage2_tools/data_tools.py`: bounded, read-only query implementation.
- `eval/stage2_tools/parser.py`: model/tool interaction and stopping conditions.
- `eval/stage2_tools/config.py`: separate method identifier and budgets.
- `eval/stage2_tools/runner.py`: fresh parser integration with Stage 2 and cost/trace recording.
- `eval/stage2_tools/collect.py`: separate collector, including tool uptake and failure counts.

```bash
.venv/bin/python notebooks/kaggle/stage2_tools/make_notebooks.py
.venv/bin/python -m pytest tests/test_stage2_tools.py tests/test_stage2_independent.py -q
```

The original profile notebooks and source hashes must remain usable. The tool notebook pack embeds only its own runner package plus shared core source and raw data. CPU tests validate the protocol, condition scopes, precision, read-only behavior, limits, error recovery and all six Stage 2 routes; actual model/tool behavior and GPU compatibility must be established on Kaggle.
