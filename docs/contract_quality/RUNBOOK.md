# Contract-quality development pilot v1

Status: CPU-validated implementation; **GPU runs have not been executed here**. This is an exploratory synthetic draft, not a reviewed selection study or a replacement for the frozen Stage 2 results.

## What is ready

- `benchmark/contract_quality_v1/`: 80 open-development tasks, 40 selection tasks, six separate ambiguous-request examples, six generated CSVs, file hashes and a review sheet.
- `eval/contract_quality/`: deterministic builder, bounded input profiles, sandboxed validation/scoring, GPU pilot and two-bundle collector.
- `results/contract_quality_v1/suite_validation.json`: all 120 supported tasks passed automated gold-contract checks. Each task has two wrong candidates, including a no-op; all 240 were rejected. Some alternate positive programs repeat the reference and are deduplicated in scoring.
- `notebooks/kaggle/contract_quality/contract_quality_pilots.zip`: two self-contained notebooks, one per saved adapter.

No existing parser, training data, adapter, frozen runner, original notebook, or Stage 2 result was changed.

## Scope and limitations before running

The four open-development datasets and two selection datasets have distinct domain-prefixed schemas and separately generated values. They **share 20 transformation templates**. This provides controlled parser diagnostics, not evidence of generalization to new task families or real-world datasets. The splits must not be described as independent benchmarks.

Coverage currently includes full-column mean/median, a conditional median, unique mode, strict/inclusive filters, similar column names, single/composite deduplication keys with first/last, ordered categories, formatted dates, casts, mapping, clipping and a scoped edit. Multi-edit tasks, keep-none, tied modes and numeric-text conversion remain gaps. Expand and review these before using the suite for final model selection or revised training design. Ambiguous examples are kept separately and are **not run/scored by this parser pilot**, whose original output protocol has no explicit escalation action.

All review rows remain `draft`. Ask an independent teammate to review requests, assumptions, CSVs, gold contracts and positive/negative programs in `review.csv`; include their name and notes. Do not mark automatic checks as human review. The pilot may run now for exploratory diagnostics, but its output cannot satisfy the reviewed-data selection gate.

The selection file is accessible to maintainers for annotation. Its lock is procedural, not encryption: model task loading refuses selection by default, and the GPU pilot and collector are hardwired to open development. Do not run or tune models on selection until the candidate and selection procedure are frozen.

## Kaggle instructions

1. Create a **private Kaggle dataset** from these existing folders, keeping their names and every adapter file:
   - `finetune/runs/qwen25_1_5b/adapter/`
   - `finetune/runs/qwen25_3b/adapter/`
   A convenient layout is `qwen25_1_5b/adapter/...` and `qwen25_3b/adapter/...`. Do not upload these weights to ordinary GitHub Git. The notebook checks the adapter configuration, weights and prompt marker against their SHA-256 hashes.
2. Extract the notebook ZIP. Upload `cq_1_5b_pilot.ipynb` into Kaggle. Attach that private dataset. Select **GPU T4 ×2** and turn **Internet on**. Only one GPU loads the parser; the second participates in the existing attention-kernel preflight. No generator or repair model is loaded.
3. Use **Save Version → Save & Run All** in a fresh session. A four-generation smoke check runs first, then **160 generations**: 80 open tasks × original/grounded input. The output directory must be empty to prevent accidentally mixing runs. Do not rerun cells into a completed output directory.
4. Wait for `160/160` and successful completion of the final cell. Download `cq_1_5b_pilot.zip` and the executed notebook. An interrupted or failed job can leave a partial ZIP; it is not a complete experiment and the collector rejects it. Save the log if any setup/runtime failure occurs.
5. Run `cq_3b_pilot.ipynb` in another fresh session with the same settings and dataset. Download `cq_3b_pilot.zip` and its executed notebook.
6. Store the two full ZIPs in `kaggle_outputs/contract_quality/v1/`. Store executed notebooks in `notebooks/executed/contract_quality/v1/`, named `cq_1_5b_pilot_executed.ipynb` and `cq_3b_pilot_executed.ipynb`. Keep template notebooks unchanged.
7. From the repository root, score both bundles locally:

```bash
.venv/bin/python -m eval.contract_quality.collect \
  kaggle_outputs/contract_quality/v1/cq_1_5b_pilot.zip \
  kaggle_outputs/contract_quality/v1/cq_3b_pilot.zip \
  --out results/contract_quality_v1/pilot_comparison.json
```

The collector requires both adapters, the correct suite/adapter hashes and all 160 records per run. It refuses smoke, incomplete, mixed-inventory and duplicate task/variant records. It runs candidate programs through the existing sandbox, never executes generated model text, and refuses to overwrite an existing report.

No GitHub push is needed: the notebook contains the relevant source and draft suite. Model weights are deliberately excluded. Inference uses the existing p1-v2 system prompt, zero-shot greedy generation, FP16 and 1,024 maximum output tokens for both input variants. The richer variant appends a deterministic full-precision input-data profile, never gold labels or reference outputs. Context overflow fails visibly rather than silently truncating information.

The runtime uses package versions from the successful Stage 2 environment. It checks Kaggle's installed torch version before loading models. If Kaggle changes its base image, share the setup error rather than silently changing packages. The historical adapters lack a pinned base revision: each pilot resolves and records a model commit for its run; historical revision reproducibility remains a limitation. GPU execution has not been tested locally, so error-free execution cannot be guaranteed.

## How to decide what comes next

Compare the paired open-task counts for correct outputs accepted, tasks with any wrong output accepted, invalid contracts, joint semantic success, error-family probes and tokens. Invalid contracts are withheld outcomes, not successful semantic validation. A weaker contract can accept more correct outputs and more wrong ones; report both.

The predeclared eventual screening criterion remains: improve correct-output acceptance without increasing observed wrong-output acceptance on the **reviewed** set, and report coverage/cost. No automatic winner is declared on this draft. After reviewing the pilot, choose the input protocol and fix coverage gaps before revising SFT data. Retraining and locked selection remain later steps; the training notebook is intentionally not produced before the grounding decision.

## CPU reproduction

```bash
.venv/bin/python -m eval.contract_quality.build_suite
.venv/bin/python -m eval.contract_quality.harness \
  --out results/contract_quality_v1/suite_validation.json
.venv/bin/python notebooks/kaggle/contract_quality/make_notebooks.py
.venv/bin/python -m pytest tests/test_contract_quality.py -q
```

The builder does not overwrite an existing review sheet. Changing suite data requires a new version, regenerated manifest/notebooks and fresh run directories. Existing frozen Stage 2 data must stay unchanged.
