# Evaluate existing selective-facts weights after the scope correction

Use `notebooks/kaggle/selective_facts_scopefix/selective_facts_scopefix_eval.ipynb` next. This runs fresh parser inference with your **existing epoch-2 1.5B adapter**. No retraining or optimizer step is performed.

## Kaggle instructions

1. Upload `notebooks/kaggle/selective_facts_scopefix/selective_facts_scopefix_source.zip` as a new **private Kaggle dataset**.
2. Upload your completed **`selective_facts_1_5b.zip`** as a second private dataset. You can use the full ZIP; there is no need to manually extract the adapter or rename its files. Kaggle's packed and unpacked dataset layouts are both supported. Use the full result, not the smoke ZIP. An already extracted dataset containing the complete original `adapter/` folder also works.
3. Create a new Kaggle notebook by importing **`selective_facts_scopefix_eval.ipynb`**. Attach both datasets, select **GPU T4 x2**, and enable **Internet**.
4. Save Version → **Run All**. No paths or model-size settings need to be edited. No original-adapter dataset or GitHub push is required.
5. Download **`selective_facts_scopefix_eval.zip`** and the executed notebook. Share them for analysis before running the six full Stage 2 notebooks.

Place the output ZIP under `kaggle_outputs/parsing/selective_facts_scopefix1/`. Place the executed notebook under `notebooks/executed/parsing/selective_facts_scopefix1/`, preferably named `selective_facts_scopefix_eval_executed.ipynb`.

This is much less work than training: it runs two smoke inference cases followed by 100 development inference cases, with CPU semantic checks. Model download and Kaggle hardware/runtime affect duration; it does not perform the prior five-hour training job. Do not run the training notebook again for this correction.

## What changed

The original scope detector treated any `when` as evidence of a row condition. The neutral sentence “Ignore missing values when computing statistics.” therefore blocked 12 full-column median, mean and mode tasks before inference.

`finetune/selective_facts_scopefix/context.py` removes only that bounded null-policy phrase **from scope detection**. It does not remove it from the user request or change the requested statistic. The existing computation already excludes missing values. Actual conditional language remains subject to the original conservative scope checks.

The new inference protocol is **`p1-v2+sf1-scopefix1`**. The input facts serialization, system instructions, request placement, NF4 loading and decoding settings are unchanged. Preflight verifies that the 12 blocked cases receive facts and the other 88 development prompts are exactly the same as SF1. Missing/ambiguous/unsupported groups, unresolved tied modes and all-missing statistics still produce explicit errors rather than invented values.

The original `finetune/selective_facts/` code, published training notebook/source bundle, raw result ZIP and adapter metadata remain untouched. The corrected evaluator records both the adapter's training protocol (`p1-v2+sf1`) and the new inference protocol. It validates the old metadata against the unchanged original code and checks that the delivered weights and metadata match the exact epoch-2 artifact from the supplied run. It never relabels the old adapter as newly trained.

## What the notebook does

- Verifies the source/data ZIP hashes and installs the pinned inference dependencies.
- Checks the Kaggle environment, the masked single-visible-GPU attention path, the 12 repaired routes, and 88 unchanged prompts.
- Finds and verifies the existing epoch-2 adapter automatically. It reads extracted input files directly, or copies the sealed adapter files from an intact ZIP to the working directory. Input datasets are never edited. The tokenizer comes from the pinned base model, as in the original run.
- Runs a two-case inference smoke (one repaired statistic case and one ordinary deduplication case).
- Loads the existing adapter in a fresh process and generates new answers on all 80 synthetic and 20 original development cases. No previous predictions are substituted for current model calls.
- Scores correct reference outputs and available targeted wrong outputs, then exports a result ZIP, logs and comparison.

`summary.json` contains the fresh results and provenance. `tasks/*.json` contains the fresh prompts, computed facts, model outputs and semantic scores. `comparison.json` includes current scores, historical original-adapter scores (68/100) and historical SF1 epoch-2 scores (73/100), per-task gains/losses, and results on the 12 previously blocked cases. The historical scores are explicitly labeled; they are not fresh control runs. `completion.json` confirms completion and `retrained: false`. Failures export partial results and `failure.json` where possible; existing result directories are refused.

## How to decide after the run

Inspect whether the 12 cases actually receive correct facts **and** produce useful contracts. Check that the other cases retain their performance and that wrong-output acceptance does not increase. The comparison includes screening flags, but neither guarantees quality nor automatically changes a Stage 2 adapter.

Do not assume the score must become 85/100: supplying a previously missing fact does not guarantee the model uses it correctly. Previously observed preservation contradictions, exact-name errors and schema-format mistakes may remain. These are open-development diagnostics, not independent test accuracy. The original development tasks have only wrong-no-op probes, whereas the synthetic tasks have targeted wrong transformations.

This notebook does not run end-to-end Stage 2 and the existing Stage 2 notebooks still use their declared protocols. Integrate the corrected protocol into a separately versioned Stage 2 workflow after reviewing the parser results. The two tool-use workflows remain separate experiments.

## Local verification and reproduction

Run `PYTHONPATH=src:. .venv/bin/python -m finetune.selective_facts_scopefix.preflight` for the CPU routing check. Rebuild the packaged notebook with `PYTHONPATH=src:. .venv/bin/python -m notebooks.kaggle.selective_facts_scopefix.make_notebook`.

Tests cover repaired statistics and precision, genuine conditional scopes, ties and empty columns, all 88 unchanged prompts, agreement with the draft gold facts in the 12 repaired cases, adapter seals and packed/unpacked discovery, historical task identity checks, a scripted two-task evaluation through the real verifier, and both Kaggle source-loader layouts. GPU inference remains to be exercised on Kaggle; no new GPU scores are claimed locally.

Implementation verification: 36 targeted tests passed across the scope-fix, original selective-facts and attention suites. The actual supplied training ZIP passed adapter discovery and all original/pinned adapter seals. The generated source ZIP passed CRC, file hashes, isolated CLI imports and the complete 100-task routing preflight. Frozen original sources passed verification. The original training source bundle and notebook were not regenerated. New GPU inference remains pending.

The existing training output is now stored locally at `kaggle_outputs/training/selective_facts_v1/selective_facts_1_5b.zip`; upload that file for the second dataset. Its original executed notebook is at `notebooks/executed/training/selective_facts_v1/selective_facts_train_executed.ipynb`. The file contents have not changed.

Completed Colab evaluation: the same epoch-2 weights pass **85/100** development references, including all **16/16** statistic-dependent cases. All 12 previously blocked tasks are recovered, with no losses versus the prior 73/100 run and zero accepted tested wrong outputs. See `results/selective_facts_analysis/scopefix1/ANALYSIS.md` for verification, remaining failures and Stage 2 integration requirements.
