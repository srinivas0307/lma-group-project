# Selective facts parser: implementation and run instructions

This version implements the proposed solution: compute only supported request-relevant facts, train a new parser with that same format, score its generated contracts, and use a screened adapter in a separate Stage 2 workflow. Its protocol is `p1-v2+sf1`; original parser weights and completed workflows are preserved.

**Run the training notebook next.** The implementation is ready for a GPU run; no new adapter has been trained locally and no quality improvement is claimed yet.

## What changed

`finetune/selective_facts/context.py` builds both training and inference inputs. It keeps the schema and first three rows, inserts needed facts before the request, and leaves the request last. It adds explicit instructions about replacement scope, exact names/keys and preservation. This prompt change is part of the new method, so a gain would not isolate facts from the added instructions or retraining.

Renames, deduplication, literal fills and other tasks receive no extra statistics. Statistic-fill requests can receive a full-precision full-column median, mean or mode. A single equality-group statistic is supported with an explicit among/within scope. For example, `Fill missing x only in zone = east with the median of x among rows with zone = east.` supplies the east-group median. `Fill missing x only in zone = east with the full-column median of x.` supplies the global median: the statistic scope and the cells to change are different.

This is a deterministic request interpreter, not LLM tool use. It requires explicit input column names. It supports a limited grammar rather than general natural language: multiple statistics, unresolved aliases, unsupported conditions, conflicting scopes, empty/all-missing groups and unresolved mode ties raise `FactUnavailable`. Mode ties can use explicit smallest/largest or alphabetically first/last rules. Recognized unsupported requests explicitly escalate before inference in the new Stage 2 workflow. Unknown natural-language constructions remain a limitation; the interpreter is not a general proof of request understanding.

The new examples teach full-column and grouped median/mean, unique mode, scoped fills using global statistics, literal fills, missing sentinels, renames, single/composite deduplication with first/last, conditional edits, filters, string casts, clipping and copying another column into missing cells. Each added gold contract accepts a constructed correct output and rejects a constructed wrong output. The added examples are automatically validated drafts without independent review; automated checks do not establish that every annotation captures every reasonable interpretation.

Original replay examples come from corrected `finetune/data/v2/checked.jsonl` and retain the original dataset-based training/validation split. Examples whose statistic information cannot be routed are excluded and listed in the new manifest. Replay examples inherit the original checked provenance; this build does not rerun every reference program. New synthetic training/validation datasets have separate names, seeds and CSVs, and neither overlaps the open development datasets. Templates share operation concepts, so these resources are not independent test evidence.

The build has **4,003 training examples and 902 validation examples**, including **540** new synthetic examples. Training has 173 examples with facts; validation has 39. Forty-five unresolved original requests are excluded. Twenty-five inherited rounded statistic labels are replaced with exact input-derived fact values in the new build, with each change logged; the original data remains intact.

The exact counts, fact/no-fact coverage, excluded IDs, precision updates, source hashes and synthetic checks are in `finetune/data/selective_facts_v1/manifest.json`.

## Compatibility correction after the first Kaggle preflight

Transformers 5.20.2 rejects the removed `warmup_ratio` argument. The CPU probe and full training now share `finetune/selective_facts/training_api.py` and use `warmup_steps=0.03`; this version interprets fractions below one as the warmup ratio. See the [pinned upstream implementation](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/training_args.py). Full training constructs the arguments before baseline inference or model loading. The constructor options and actual training arguments are recorded in run metadata.

If you encountered that error, replace the private source/data dataset with the regenerated ZIP **and** re-import the regenerated small notebook below. Their source-manifest hashes must match. Keep the original adapter dataset attached. Start a fresh session and run all; the failed preflight did not start the full training job. The new constructor regression tests run locally without Torch; the actual tiny CPU training probe and GPU job still require Kaggle verification.

## GPU isolation correction after the baseline smoke failure

Training and semantic evaluation each expose one physical GPU through `CUDA_VISIBLE_DEVICES`. In either subprocess the visible card has local index `cuda:0`. The old frozen attention helper tested `[0, 1]` unconditionally, so it crashed in `fork_rng` before baseline inference when only local device 0 existed.

The sf1 paths now use `finetune/selective_facts/attention.py`, which derives RNG and self-test indices from `torch.cuda.device_count()`. It reuses the existing efficient-attention/GQA implementation without changing the frozen helper. Failed tests restore the original attention function. Training and evaluation import this new helper; their model placement remains local `cuda:0` on their respective physical cards.

The notebook now tests attention forward numerics and backward gradients in two separate single-visible-GPU subprocesses, one per physical T4, before launching the smoke or full job. Local regression tests cover both one-GPU masks, two visible GPUs, invalid devices, and failure cleanup without requiring local CUDA; actual GPU kernel execution remains a Kaggle preflight check.

After that failure, replace **both** the source/data ZIP and small notebook with the regenerated files, attach the latest source-dataset version, and start a fresh session. The failure occurred during the smoke baseline; full fine-tuning had not started. The unauthenticated Hugging Face warning is separate from this CUDA-index failure and does not require a token for these public model downloads.

## What you need to do now

1. Upload **`notebooks/kaggle/selective_facts/selective_facts_training_source.zip`** as a new **private Kaggle dataset**. Keep its contents unchanged; the notebook supports either an intact ZIP or automatically extracted files.
2. Upload the updated small **`notebooks/kaggle/selective_facts/selective_facts_train.ipynb`** to a new Kaggle notebook.
3. Choose **GPU T4 x2**, enable **Internet**, and attach **both** the new source/data dataset and your original 1.5B adapter dataset at `/kaggle/input/datasets/srinivasu2005/qwen-1-5b-adapter`. Keep all original adapter files together. This adapter supplies the baseline comparison; training starts a fresh adapter from the pinned base. You do not need the 3B adapter, earlier result ZIPs or a GitHub push.
4. Use **Save Version → Save & Run All** in a fresh session. The notebook checks optional-package compatibility, runs a tiny CPU Trainer/LoRA/save/reload probe, runs two training steps as an infrastructure smoke job, then starts a separate full two-epoch job automatically.
5. Wait for **COMPLETE: training, full validation loss and semantic checkpoint evaluation.** Download **`selective_facts_1_5b.zip`** and the executed notebook.
6. Put the ZIP in **`kaggle_outputs/training/selective_facts_v1/`**, and the executed notebook in **`notebooks/executed/training/selective_facts_v1/`**. Share those two files for analysis.

The updated notebook is small; its source/data are supplied by the sealed private-dataset ZIP. The previous embedded version was 9.6 MB and was replaced after a Kaggle CommitAndRun 499 failure. The large source payload is a plausible contributor, not a confirmed diagnosis of that service error. The old embedded template is archived under `archive/notebooks/training/selective_facts_embedded/`.

The loader checks the exact manifest hash and every source/data file hash, supports zipped and automatically extracted Kaggle datasets, and rejects missing or changed inputs. The bundle contains prepared training inputs and only open development annotations, with no previous model outputs or weights. The notebook includes Python language metadata and normal readable setup cells. Model weights download from Hugging Face. Kaggle must provide the same Torch `2.11.0+cu130` environment used in the preceding successful diagnostic; a preflight fails visibly on a different environment.

Training sees only GPU 0, avoiding accidental multi-GPU DataParallel behavior. GPU 1 runs semantic evaluation subprocesses while training waits. QLoRA uses NF4/FP16 computation, rank 16, alpha 32, two epochs, batch 1 with accumulation 16, maximum 4096 tokens and fixed base/tokenizer revision. Neither training nor inference uses double quantization in this version. Prompt tokens and padding are masked from loss; assistant contract tokens are supervised. Oversized examples cause a preflight error instead of being silently truncated or dropped. Full validation loss uses every designated validation example in the full job.

## How checkpoint evaluation works

The original adapter first generates contracts in its original input format. After each epoch, the new adapter generates contracts in the sf1 format. Each is evaluated on the same 80 synthetic open-development tasks plus 20 existing development tasks. No original test or locked-selection task is used for model selection. Synthetic tasks have targeted wrong programs plus no-op probes; original development tasks have only wrong-no-op probes, a weaker check. Correct reference outputs never enter parser prompts.

For each cohort, a candidate must accept at least as many correct outputs and accept no more wrong outputs than the original control, while improving total joint successes (correct accepted and none of the supplied wrong outputs accepted). This is finite-sample screening, not a guarantee about unseen requests. A candidate can complete training and still fail screening.

The ZIP contains `adapter/`, `run_info.json`, `train_log.jsonl`, `baseline/`, `semantic/`, `selection.json` and `completion.json`. Every evaluated epoch's predictions and scores are kept, but intermediate weight copies are omitted from the ZIP to reduce size. The best candidate is retained even if none is eligible, with `eligible: false` recorded. Actual optimizer steps, versions, token lengths, runtime, dataset/source hashes and effective batch size are recorded. Smoke artifacts never qualify for Stage 2. On failure, a partial sealed ZIP and `failure.json` are exported.

## After we inspect the training result

If `selection.json` reports `eligible: true`, create a **new private Kaggle dataset** containing the complete new `adapter/` directory, including `input_protocol.json`, `selection.json` and tokenizer files. Do not replace the old adapter dataset. The runtime checks protocol, revision, the exact facts implementation and hashes binding the adapter to its evaluated checkpoint.

Then run **`notebooks/kaggle/stage2_selective/stage2_selective_05_proposed.ipynb`** first with T4 x2, Internet and that new adapter attached. It computes fresh contracts from CSVs and requests within its own job; it does not consume training predictions or earlier Stage 2 outputs. Full output is `stage2_selective_proposed.zip`.

The complete later comparison is packaged in `notebooks/kaggle/stage2_selective/stage2_selective_notebooks.zip`: six independent runs plus `stage2_selective_collect.ipynb`. Only proposed and no-contract-to-generator need the new adapter. The prompted 7B arm keeps p1-v3 and receives the new facts rules; conventional thresholds remain hand-set. Collect only sf1 bundles with the matching collector. Store results in `kaggle_outputs/stage2/selective_facts_v1/` and executed notebooks in `notebooks/executed/stage2/selective_facts_v1/`.

If no checkpoint passes screening, share the training ZIP before another full Stage 2 run. The artifact is still useful for diagnosing which operations, scopes or literals need better training data. The old 100 test tasks have already informed development; reruns are diagnostic, not fresh confirmatory evaluation.

## Local verification and maintenance

- Tests: `.venv/bin/python -m pytest tests/test_selective_facts.py -q`.
- Data build: `.venv/bin/python -m finetune.selective_facts.build_data --out finetune/data/<new_version>`. It refuses to overwrite a build. Update the notebook pack deliberately if creating a different dataset version.
- Training template: `.venv/bin/python notebooks/kaggle/selective_facts/make_notebooks.py`.
- Stage 2 templates: `.venv/bin/python notebooks/kaggle/stage2_selective/make_notebooks.py`.

CPU tests verify facts, scope distinctions, precision, ambiguity behavior, training/inference agreement, loss masking, dataset separation, screening, fresh six-arm wiring and source/data bundle loading in both Kaggle layouts. Local Torch/Transformers are unavailable, so real GPU training, the Kaggle CPU Trainer probe and model-quality improvements remain to be verified by the supplied notebook.
