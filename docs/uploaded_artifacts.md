# Uploaded notebooks and Kaggle results

The six user-uploaded files formerly at the project root have been moved without changing their contents. Executed notebooks have descriptive names. Result ZIP names are retained for Kaggle compatibility. `uploaded_artifacts.json` records the original names, current paths and SHA-256 hashes.

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

The earlier independent-v1 proposed run's ZIP and executed notebook are not among these six uploads. Its analyzer now expects their proper independent-v1 folders rather than the filename later reused for the diagnostic.

## Where to save future runs

Keep editable/upload-ready notebooks in `notebooks/kaggle/`. Put downloaded executed notebooks in `notebooks/executed/<workflow>/<version>/`, and downloaded result ZIPs in `kaggle_outputs/<workflow>/<version>/`. Include model size where relevant. Avoid replacing an existing completed run; use a distinct run name or subfolder.

For the next scope-fix evaluation specifically:

- ZIP: `kaggle_outputs/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval.zip`.
- Executed notebook: `notebooks/executed/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval_executed.ipynb`.

The corresponding analysis scripts have been updated to use the organized locations. No notebooks or result ZIPs were deleted.
