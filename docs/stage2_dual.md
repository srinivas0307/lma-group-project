# Stage 2: selective facts versus trained tools

The new implementation is `eval/stage2_dual/`. It provides two **independent proposed-system runs** and a CPU comparison collector. Both runs start from the raw requests/CSV files and generate fresh contracts and code. They do not consume previous parser evaluation ZIPs, Stage 1 parses, generated code, tuning outputs or each other's results. No training is performed.

## Run on Colab

Use your existing **`MyDrive/igt_scopefix`** folder, as in the working parser notebooks. The first notebook cell lets you change this location if needed.

1. Put the new **`notebooks/kaggle/stage2_dual/stage2_dual_source.zip`** in that folder.
2. Keep the existing **`selective_facts_1_5b.zip`** there. This is the full selective-facts training output, locally at `kaggle_outputs/training/selective_facts_v1/selective_facts_1_5b.zip`.
3. Open **`notebooks/colab/stage2_dual/stage2_dual_selective_proposed_colab.ipynb`**, enable a T4 GPU and Internet access, then Run all.
4. It runs a two-task infrastructure smoke, then all 100 test tasks. Download **`stage2_dual_selective_proposed.zip`** and the executed notebook. Full and partial ZIPs/logs are copied to `MyDrive/igt_scopefix/stage2_dual_results_<timestamp>/` where possible.
5. In a **separate Colab session**, put **`tool_use_train_1_5b.zip`** in the same Drive input folder. This is your full tool-training output, locally at `kaggle_outputs/tool_use_v2/tool_use_train_1_5b.zip`.
6. Open **`notebooks/colab/stage2_dual/stage2_dual_tools_proposed_colab.ipynb`** and Run all with a T4 GPU. It needs only the source ZIP and its own tool-training ZIP. Download **`stage2_dual_tools_proposed.zip`** and the executed notebook.
7. Open **`notebooks/colab/stage2_dual/stage2_dual_collect_colab.ipynb`** on CPU. Keep the source ZIP and the two full Stage 2 output ZIPs under `MyDrive/igt_scopefix` (the timestamped result subfolders are searched). Run all. Download **`stage2_dual_comparison.zip`** and the collector notebook.

You can run the two GPU notebooks in parallel in independent sessions if your available quota permits. Neither waits for the other. The collector is the only step that needs both outputs. It requires matching actual software/Python versions, so use matching environments for both model runs. The actual package/Torch/CUDA/GPU details are recorded in each result; do not compare a different Torch/Python setup silently.

No paths need changing when you use the existing Drive folder and exact ZIP names. Both intact ZIPs and unpacked adapter/source datasets are supported. Each adapter is verified against the selected epoch-2 weights and original metadata. Old adapters, smoke adapters, the other method's adapter and different checkpoints are rejected. The original metadata is never rewritten.

## Run on Kaggle instead

The corresponding notebooks are in `notebooks/kaggle/stage2_dual/`:

- `stage2_dual_selective_proposed_kaggle.ipynb`: attach `stage2_dual_source.zip` and `selective_facts_1_5b.zip` as private datasets.
- `stage2_dual_tools_proposed_kaggle.ipynb`: attach `stage2_dual_source.zip` and `tool_use_train_1_5b.zip`.
- `stage2_dual_collect_kaggle.ipynb`: CPU; attach the source ZIP and the two full output ZIPs.

Enable Internet. GPU notebooks support T4 and T4 x2; only one model is resident at any time. Choose the same software environment for both. Save Version → Run All. Download the full result ZIPs and executed notebooks. The shared source ZIP is small; model weights are separate inputs/downloads, not embedded into notebook cells.

## What each method uses

**Selective facts:** exact selected epoch-2 Qwen2.5-1.5B selective-facts adapter; corrected scope routing; supported request-relevant input facts; narrow recognized-operator whitespace normalization; one parser answer; **no model-correction branch**. Protocol: `p1-v2+sf1-scopefix1-norm1`.

**Tools:** exact selected epoch-2 Qwen2.5-1.5B tool adapter; schema/sample plus the existing tool instructions; real read-only `profile_column`, `count_rows`, `inspect_rows`, `group_profile`; at most six tool attempts, ten model replies, two invalid-response corrections, 24,000 cumulative parser tokens and 12,288 parser context tokens. Protocol: `parser-tools-v2+norm1`. All queries and failures, including `__unknown_column__` and invented tool names, remain in the logs. There is no forced data query, hidden substitution of tool arguments or unmeasured extra model-correction layer.

Normalization is applied only to prospective final contract replies, after checking for tool/escalation actions. It removes surrounding whitespace from recognized operators at schema positions, logs every change and preserves the raw reply. Tool arguments, column names, literal values, unknown fields and tolerances are not silently rewritten. Applying it inside the tool loop can avoid an invalid-response retry, so a fresh Stage 2 run is required; the offline 91/100 tool result is not assumed to carry over exactly.

Both use the **same** pinned 3B generator, 7B NF4 repair agent, generator/repair prompt implementations, verifier, sandbox and limits: three repair proposals, eight repair tool calls per attempt, 60,000 repair tokens, 512 maximum initial-generation tokens and 768 maximum tokens per repair reply. The repair prompt remains `a1-v4`; generator prompt identity/hash is recorded. The proposed arm uses the existing contract-aware `b2-v2` generator. This is a newly versioned pipeline, so do not silently merge it with earlier frozen b1/b2 results.

The parser, generator and agent base revisions are pinned in `eval/stage2_dual/config.py`. The two adapters have different training conversations/prompts; this compares complete parser methods, not a controlled ablation of tool access alone. Parser budgets/costs differ by method and are reported explicitly.

## Single-GPU memory strategy

The runner loads **one model at a time**:

1. Load the parser, generate every task's contract, save complete traces, unload it.
2. Load the 3B generator, generate each eligible task's initial code, save its exact response/token counts, unload it.
3. Load the 7B repair agent, execute/verify the current run's initial code and repair when needed, then evaluate against the reference.

The generated-code responses are buffered **within the same fresh run**. The repair phase reuses exactly that one fresh response per task, with an assertion that the generator prompt matches. It does not call the generator twice or reuse another notebook's outputs. Recorded costs charge that initial generation once. On one GPU, all roles use GPU 0; on two GPUs the generator uses GPU 1 and parser/repair use GPU 0, while remaining sequential.

GPU preflight tests the actual attention implementation before loading model weights. This strategy avoids simultaneous parser/generator/repair model residency; it cannot guarantee freedom from platform interruptions, quota expiry or every GPU failure. All declared context limits remain explicit. A parser context refusal escalates; a repair context refusal is treated as a declared budget stop by the unchanged repair loop and recorded in the plan. CUDA or model-loading failures fail the job and export partial diagnostics rather than being presented as ordinary quality outcomes.

## Smoke and full run

The two-task smoke uses `adult_col_001` (ordinary edit) and `titanic_miss_004` (input statistic). It performs parsing, initial generation, verification/repair and reference scoring with real models. Its purpose is infrastructure validation; ACCEPT, REJECT or ESCALATE are legitimate smoke outcomes. After it finishes, the full 100-task run starts afresh in a separate output directory, with fresh model responses.

Existing output directories are refused. Partial sealed ZIPs are exported during parsing/generation and after each evaluated task, and on handled errors. Colab copies the available ZIPs/logs to Drive when a child run returns/fails. A forced platform kill may prevent the last copy; no automatic resume behavior is claimed. This delivery uses a full run per notebook, not a separate resume notebook.

## What the collector validates and reports

The collector checks full ZIP seals/coverage, method/protocol, same source/runtime, task and raw-input inventory, exact adapter seals, declared model revisions, generator/repair prompt identity, matching Python/package versions, fresh contracts and initial-generation records, outcomes and parser/tool/token accounting. Smoke/scripted/partial, mixed-source, wrong-adapter and missing/duplicated records are rejected. It loads no model weights and performs no training.

The report includes:

- Correct accepted transformations, incorrect accepted transformations, withholding and parse escalations.
- Initially correct outputs subsequently lost, and initially wrong outputs repaired.
- Contracts rejecting a correct reference, parser calls/tokens, actual tool attempts/failures and normalized task counts.
- Generation/repair tokens and total tokens.
- Per-task correctness differences between the two methods.

Per-task times include parser and generation/repair execution but exclude model loading/setup; they are not full notebook wall-clock durations. Hardware and environment are recorded, so interpret time comparisons accordingly. The collector validates saved runner outputs and provenance; it does not independently reexecute all reference programs. A later CPU failure audit can replay the traces as with earlier experiments.

## Where to put downloads

Save full result ZIPs under **`kaggle_outputs/stage2/dual_parser_v1/`**:

- `stage2_dual_selective_proposed.zip`
- `stage2_dual_tools_proposed.zip`
- `stage2_dual_comparison.zip`

Save executed notebooks under **`notebooks/executed/stage2/dual_parser_v1/`**, including method and platform in filenames. Share the two full run ZIPs, collector ZIP and executed notebooks for failure analysis. Keep repeated experiments in separate run subfolders rather than overwriting a completed run.

## Scope of this delivery

There are **two proposed-system notebooks and one collector per platform**. These are the initial comparison described in the plan. The underlying runner retains the six preset routes (`baseline1`, `baseline2_repair`, `baseline3_repair`, `gold_contract`, `proposed`, `no_contract_to_generator`) and the collector can collect a same-method six-system set with `--six`. Separate six-system notebook packs can follow after reviewing the proposed-method results; this delivery does not require rerunning all twelve system/method combinations upfront.

Only the explicit gold-contract preset can pass a gold contract to generation/repair. Ordinary methods receive request/schema/sample and permitted input facts/tool observations. Reference programs/outputs are used only after that task's runtime has finished. No gold result is used to choose a contract or repair action.

The existing 100 test tasks have already informed development repeatedly. Treat these runs as diagnostic end-to-end comparisons. A stronger generalization claim still needs a genuinely independent benchmark.

## Implementation checks

Rebuild with `PYTHONPATH=src:. .venv/bin/python -m notebooks.kaggle.stage2_dual.make_notebooks`. CPU routing/protocol preflight: `PYTHONPATH=src:. .venv/bin/python -m eval.stage2_dual.preflight`.

Tests exercise normalized selective facts, the genuine tool-observation loop with retained errors, one-model residency, within-run generation accounting, successful acceptance, real repair through the unchanged sandbox/verifier, parser failures, explicit context limits, strict collection/corruption rejection and both packed/unpacked source layouts on both platforms. Source ZIP extraction/CLI checks and both actual selected adapter ZIPs/folders are verified locally. Actual full GPU Stage 2 quality remains to be measured; scripted CPU tests are not model-quality results.

Implementation validation completed: **24 tests passed** (`tests/test_stage2_dual.py` and `tests/test_selective_attention.py`). This includes all six retained preset routes, both proposed methods, real repair execution, explicit context refusal, missing-generation rejection and both platforms' source loaders. The final sealed source ZIP passed isolated CLI/preflight and exact runtime-hash comparison. Both real selected adapters validated from intact ZIPs and extracted folders. Original frozen source checks passed. Full GPU Stage 2 results remain pending.
