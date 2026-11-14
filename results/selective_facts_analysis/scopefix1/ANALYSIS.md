# Selective-facts scope-fix evaluation

**The correction worked on all 12 previously blocked cases. The existing epoch-2 adapter now passes 85/100 parser-development reference checks, up from 73/100, with no lost passes and no accepted tested wrong outputs.** No retraining was performed.

## Results

- Original contract adapter, historical original input: **68/100** correct references, four tasks with accepted wrong probes.
- Selected selective-facts adapter with the old SF1 router: **73/100**, zero tasks with accepted wrong probes.
- The **same adapter weights** with `p1-v2+sf1-scopefix1`: **85/100**, zero tasks with accepted wrong probes.
- Synthetic development: **56/80 → 68/80**; original development: **17/20 → 17/20**.
- All **12** newly supplied full-column median, mean and mode cases pass the correct reference and reject their tested wrong outputs.
- Across all statistic-dependent synthetic cases, including the four already working grouped medians: **16/16** pass.
- The other **88** cases have exactly the same prompts, raw model outputs, predicted contracts and semantic results as the prior epoch-2 evaluation. There are no lost cases versus SF1.
- Schema-valid contracts: **94/100**. Six model outputs remain invalid; nine additional schema-valid contracts do not accept the correct reference.

These are intent-parser development checks, not full Stage 2 accuracy. The synthetic suite is an automatically checked draft without independent review, and original development cases have only a wrong-no-op probe. Zero accepted tested wrong outputs is not a guarantee against all wrong transformations. These tasks have been used for development and checkpoint selection, so the score is not an independent research test result.

## What this establishes

The earlier twelve failures were caused by the implemented scope router stopping before inference. After the bounded correction, the adapter uses the supplied full-column median, mean and mode facts successfully in all twelve cases. The facts come from the input CSV, not gold contracts or reference outputs.

The run reuses the exact selected epoch-2 adapter (`adapter_model.safetensors` SHA-256 `b1adae381d17e95d1eb9abafedcac2d42b1aff57bfe2681b834060901a1826e7`). The training protocol remains `p1-v2+sf1`; the inference protocol is explicitly versioned as `p1-v2+sf1-scopefix1`. The original adapter metadata was not rewritten. `completion.json` confirms `complete: true`, `smoke: false`, `retrained: false`.

This supports using selective input facts with the matching trained parser for the supported statistic requests in this suite. It does not establish unrestricted natural-language scope handling, native tool calling, or end-to-end repair gains. The parser is automatically supplied selective facts; it does not choose a tool call.

## Remaining 15 failures

The remaining errors are unchanged from the prior epoch-2 run:

- **Four mapping tasks:** the contract requests category mapping but simultaneously requires original values to remain unchanged. Correct mappings are rejected.
- **Four arithmetic edits:** contracts confuse changing an existing column with adding a new column, sometimes inventing columns such as `add`. Two produce UNCERTAIN on a correct output because the existing column is declared newly added.
- **Three numeric filters:** the model emits the invalid comparison operator `"> "` with a trailing space.
- **One datetime conversion:** the contract includes an unsupported `format` field.
- **One Adult rename:** it changes the exact input name `education-num` to `education_num`.
- **One Airbnb filter:** it selects `neighbourhood` instead of `neighbourhood_group`.
- **One Airbnb copy-fill:** the identity relation omits its source argument and fails schema validation.

This yields six invalid contracts and nine schema-valid but reference-incompatible contracts, for 15 withheld correct references. Withholding is a verifier refusing to endorse the output under the generated contract; these are not 15 attempted Stage 2 repair failures.

Relative to the original adapter's 68/100 development score, this candidate gains 24 cases and loses seven, net +17. The seven historical regressions are the four mappings and three filter-operator cases. The scope correction introduces none of these; it preserves all 88 previously generated nonblocked responses. The overall improvement should not obscure those persistent regressions.

## Notebook and environment

The uploaded notebook is a **Colab adaptation** using Google Drive for source/adapter inputs and output copies. Its cells have no exception outputs, the smoke job completed, and the full job completed all 100 cases. Its source/data manifest SHA is the published `21063deacc439e196c07bad84cf4b19ba25b99d2cf8f85a49587d457e7bb0e9b` bundle. Runtime and input hashes validate.

The launcher uses Colab's installed Torch/CUDA build and a single visible GPU instead of the Kaggle-specific Torch/T4-x2 assertion. It writes `colab_environment.json` to Drive, but that file is not included in this supplied result ZIP. Exact Colab Torch/CUDA/GPU versions cannot therefore be independently read from this bundle. This limits a hardware/environment comparison; it does not invalidate the CPU-verified semantic results. All 88 previously generated outputs nevertheless match byte-for-byte across the two runs.

Summed recorded model inference time is about **25.5 minutes**, excluding setup, download/model loading, smoke and CPU semantic checking. Total tokens are **166,045**, up from 145,496 because the twelve cases now actually invoke the model. Comparing the old incomplete routing cost to this run is not a controlled efficiency comparison.

## Next step

**Keep this adapter and test the corrected parser in the proposed end-to-end Stage 2 arm next.** No additional selective-facts training is needed to evaluate that integration. The existing Stage 2 notebooks still declare the original SF1 implementation, so they must be explicitly versioned/wired to use the corrected scope router before running; simply uploading these weights into those notebooks would not apply this fix.

Run the proposed arm first and inspect correct acceptance, incorrect acceptance, withholding and repair behavior. If it is worthwhile, continue with the six-system comparison. Treat reruns on the previously inspected 100 test tasks as diagnostic; an independent benchmark is still needed for a stronger generalization claim. Keep the currently running 1.5B/3B tool-training experiments separate and compare their completed adapters when available.

The remaining mapping, arithmetic, exact-name and format errors are candidates for later targeted improvements. This result gives a reason to move to end-to-end evaluation rather than immediately spending another training run.

## Audit

Reproduce with `.venv/bin/python tools/analyze_selective_facts_scopefix.py`.

The audit verifies bundle seals, source/input hashes, exact epoch-2 adapter seals, full coverage, summary/comparison arithmetic, reconstructed facts/messages and extraction. It performs fresh CPU semantic replay of the 12 newly generated cases. For the remaining 88, it binds exactly unchanged inputs, prompts, predictions and scores to the earlier clean full CPU audit, whose hash is recorded; it does not claim to rerun those same sandbox executions unnecessarily. The detailed report is `audit.json`.

## Organized upload locations

- Result ZIP: `kaggle_outputs/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval.zip`.
- Executed notebook: `notebooks/executed/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval_colab_executed.ipynb`.

Only locations/names changed; uploaded contents remain intact.
