# Uploaded notebooks and Kaggle results

User-uploaded files formerly at the project root have been moved without changing their contents. Executed notebooks have descriptive names. Result ZIP names are retained for Kaggle compatibility. `uploaded_artifacts.json` records the original names, current paths and SHA-256 hashes.

## Selective-facts training

- Executed notebook: [selective_facts_train_executed.ipynb](../notebooks/executed/training/selective_facts_v1/selective_facts_train_executed.ipynb) — formerly `stage2-selective.ipynb`.
- Training output and selected adapter: [selective_facts_1_5b.zip](../kaggle_outputs/training/selective_facts_v1/selective_facts_1_5b.zip).
- Analysis: [selective-facts report](../results/selective_facts_analysis/v1/ANALYSIS.md).

Use this training-output ZIP as the existing-adapter dataset for the new scope-fix evaluation notebook.

## Tool-enabled Stage 2, proposed system

- Executed notebook: [stage2_tools_05_proposed_executed.ipynb](../notebooks/executed/stage2/tools_v1/stage2_tools_05_proposed_executed.ipynb) — formerly `stage2-tool.ipynb`.
- Output: [stage2_tools_proposed.zip](../kaggle_outputs/stage2/tools_v1/stage2_tools_proposed.zip).
- Analysis: [tool-run report](../results/stage2_tools_analysis/proposed_v1/ANALYSIS.md).

## Parser diagnostic

- Executed notebook: [stage2_parser_diagnostic_executed.ipynb](../notebooks/executed/stage2/independent_v2/stage2_parser_diagnostic_executed.ipynb) — formerly `stage2-independent-05.ipynb`. Its contents are the parser diagnostic, despite the original name.
- Output: [stage2_parser_diagnostic.zip](../kaggle_outputs/stage2/independent_v2/stage2_parser_diagnostic.zip).
- Analysis: [parser-diagnostic report](../results/stage2_independent_analysis/parser_diagnostic_v1/ANALYSIS.md).

The earlier independent-v1 proposed run's ZIP and executed notebook are not among those original six uploads. Its analyzer now expects their proper independent-v1 folders rather than the filename later reused for the diagnostic.

## Where to save future runs

Keep editable/upload-ready notebooks in `notebooks/kaggle/`. Put downloaded executed notebooks in `notebooks/executed/<workflow>/<version>/`, and downloaded result ZIPs in `kaggle_outputs/<workflow>/<version>/`. Include model size where relevant. Avoid replacing an existing completed run; use a distinct run name or subfolder.

For the next scope-fix evaluation specifically:

- ZIP: `kaggle_outputs/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval.zip`.
- Executed notebook: `notebooks/executed/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval_executed.ipynb`.

The corresponding analysis scripts have been updated to use the organized locations. No notebooks or result ZIPs were deleted.

## Direct base-model tools, 1.5B and 3B

The latest two notebook filenames were swapped relative to their actual `MODEL_SIZE` values. They have been organized by verified actual size:

- 1.5B notebook: [tool_use_base_1_5b_executed.ipynb](../notebooks/executed/tool_use_v2/tool_use_base_1_5b_executed.ipynb), uploaded as `stage2-tools-nf-3b.ipynb`.
- 3B notebook: [tool_use_base_3b_executed.ipynb](../notebooks/executed/tool_use_v2/tool_use_base_3b_executed.ipynb), uploaded as `stage2-tools-nf-15b.ipynb`.
- 1.5B output: [tool_use_base_1_5b.zip](../kaggle_outputs/tool_use_v2/tool_use_base_1_5b.zip).
- 3B output: [tool_use_base_3b.zip](../kaggle_outputs/tool_use_v2/tool_use_base_3b.zip).
- Analysis: [direct-tool comparison](../results/tool_use_v2_analysis/base_comparison/ANALYSIS.md).

Original contents and hashes are preserved. The ZIP labels already matched their actual model sizes.

## Completed selective-facts scope-fix evaluation (Colab)

- Output: [selective_facts_scopefix_eval.zip](../kaggle_outputs/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval.zip).
- Executed notebook: [selective_facts_scopefix_eval_colab_executed.ipynb](../notebooks/executed/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval_colab_executed.ipynb), uploaded as `selective_facts_scopefix_eval_colab.ipynb`.
- Analysis: [scope-fix evaluation report](../results/selective_facts_analysis/scopefix1/ANALYSIS.md).

The uploaded notebook uses Colab/Drive; it is preserved separately from the Kaggle upload-ready launcher.

## Completed 1.5B tool-use training

- Training output and selected adapter: [tool_use_train_1_5b.zip](../kaggle_outputs/tool_use_v2/tool_use_train_1_5b.zip).
- Executed notebook: [tool_use_train_1_5b_executed.ipynb](../notebooks/executed/tool_use_v2/tool_use_train_1_5b_executed.ipynb), uploaded as `tool_use_train_executed.ipynb`.
- Analysis: [1.5B tool-training report](../results/tool_use_v2_analysis/train_1_5b/ANALYSIS.md).

The notebook's actual size is 1.5B. Its original contents and the full output ZIP are preserved.

## Completed selective-facts correction comparison

- Output: [selective_facts_correction_eval.zip](../kaggle_outputs/parsing/selective_facts_correction1/selective_facts_correction_eval.zip).
- Executed notebook: [selective_facts_correction_colab_executed.ipynb](../notebooks/executed/parsing/selective_facts_correction1/selective_facts_correction_colab_executed.ipynb), uploaded as `selective_facts_correction_colab(1).ipynb`.
- Analysis: [completed correction comparison](../results/selective_facts_analysis/correction1/completed/ANALYSIS.md).

Fresh paired scores: 85 control, 88 normalization, 88 one-correction; zero accepted tested wrong outputs in all arms. Normalization-only is the recommended next Stage 2 integration candidate.
