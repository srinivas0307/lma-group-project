# Combined 3B with gold contracts

This is an implemented oracle diagnostic, `stage2-combined-gold-3b-fp16-v1`. Completed GPU result: **84 correct, zero incorrect accepted and 16 withheld**; see [the audited analysis](../results/stage2_combined_gold_analysis/ANALYSIS.md). The original notebook hits a JSON mapping-key comparison bug after inference completes. Use `tools/analyze_stage2_combined_gold.py` for offline collection of the saved output; no GPU rerun is needed. The frozen source/notebook are retained for provenance. It replaces predicted contracts with each task's gold contract in both generation and verification. It keeps the completed combined workflow's pinned 3B FP16 generator, 3B FP16 repair model, combined evidence/recovery loop, schema guard, prompts, sandbox, data and budgets. The completed experiment sources and notebooks are preserved.

One gold run serves both comparisons because selective facts and tools share the same downstream architecture. No parser model or adapter is loaded. Parser calls/tokens are zero; generation and repair costs are reported. Reference code is used for evaluation after inference and is not included in model prompts.

## Kaggle instructions

1. Upload `notebooks/kaggle/stage2_combined_gold/stage2_combined_gold_source.zip` as a private Kaggle dataset. Kaggle may extract it automatically; both intact and extracted forms are supported. No adapter datasets or previous outputs are required.
2. Import `notebooks/kaggle/stage2_combined_gold/stage2_combined_gold_3b_kaggle.ipynb` into Kaggle and attach that source dataset.
3. Enable Internet and select **GPU T4 x2**, matching the completed predicted-contract runs. Use **Save Version → Run All**. The notebook installs the matching pinned dependencies and checks attention/runtime infrastructure before inference.
4. It performs two smoke tasks, 20 development tasks, then 100 diagnostic test tasks. Each is a fresh job with its own output folder and ZIP. A smoke task may accept, reject or escalate; the smoke checks infrastructure, not accuracy.
5. Download **`stage2_combined_gold_3b.zip`**, **`stage2_combined_gold_summary.zip`**, and the executed notebook. They appear inside `/kaggle/working/run_<timestamp>/`. Development and smoke bundles have `_dev` and `_smoke` suffixes; the main analysis needs the full 100-task ZIP.

The summary reports correct accepted outputs, incorrect acceptances, withheld tasks, repairs, initially correct outputs lost, contract/reference compatibility, calls and costs. Failed jobs export partial bundles and logs, but these are not accepted as full results. No GitHub push or retraining is required.

## Optional comparison in the same notebook

Attach the existing completed result ZIPs as **separate Kaggle datasets** to avoid overlapping directory names:

- `kaggle_outputs/stage2/combined_3b_v1/stage2_combined_3b_selective_proposed.zip`
- `kaggle_outputs/stage2/combined_3b_v1/stage2_combined_3b_tools_proposed.zip`

These are used only by the optional final comparison cell, after gold inference finishes. They do not affect the gold run. Intact ZIPs and Kaggle-extracted sealed folders are supported. Neither uploaded result is required to run the gold experiment.

The comparison validates the original experiment's exact packaged source, settings, seals and task inventory. Gold and predicted runs must match inputs, model revisions/precision policy, combined repair settings, generation prompt, budgets and environment. Mismatches stop the comparison rather than produce a misleading matched result; the independently completed gold ZIP remains available. In that case, download the gold bundle for analysis alongside the old results, with the mismatch reported.

When both predicted bundles are attached, the final cell writes `stage2_combined_gold_comparison.zip`. Its JSON includes task IDs correct only with gold and task IDs correct only with each predicted method, alongside all summaries. The notebook also works with just one predicted bundle.

## Interpretation

Compare the new measured gold result against the completed predicted results, each 74 correct, one incorrect accepted and 25 withheld. Do not substitute the older 94/100 gold score for this new configuration's result.

Gold contracts change the generator's input as well as checking/repair, so initial programs can differ. The difference measures the total effect of replacing the contract source under the same downstream architecture; it is not a repair-only comparison. A gold score is an oracle diagnostic, not a guaranteed ceiling or deployable performance. Any remaining failures can involve generation, repair, contract expressiveness or verifier behavior and need per-task examination. These previously inspected tasks provide diagnostic evidence, not a fresh independent test.

## Local implementation and reproduction

- `eval/stage2_combined_gold/`: separate runner, runtime boundary, sealed artifacts and strict collector.
- `notebooks/kaggle/stage2_combined_gold/make_notebook.py`: builds the new pack from the exact original combined pack plus new modules; does not rebuild or edit the old pack.
- `tests/test_stage2_combined_gold.py`: scripted CPU checks for parser bypass, fresh generation, guard/repair behavior, contract authenticity, paired comparison and intact/extracted notebook inputs.

From the extracted source folder, with its pinned dependencies and source/src on PYTHONPATH, run:

```bash
python -m eval.stage2_combined_gold.runner --method gold --preset gold_contract --out /kaggle/working/gold_fresh --packages /kaggle/working/stage2_combined_gold_packages
python -m eval.stage2_combined_gold.collect --gold /kaggle/working/gold_fresh.zip --out /kaggle/working/gold_summary_fresh
```

For a paired comparison, add `--predicted <selective.zip> <tools.zip>` to the collector command. All output directories must be fresh.
