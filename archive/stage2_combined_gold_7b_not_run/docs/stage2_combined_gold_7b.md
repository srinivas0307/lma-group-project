# Combined gold-contract workflow with 7B NF4 repair

Implemented experiment: `stage2-combined-gold-7b-nf4-v1`. CPU validation passed; GPU results are pending.

This version uses **gold contracts**, a **Qwen2.5-3B-Instruct FP16 code generator**, and a **Qwen2.5-7B-Instruct NF4 repair model with float16 computation**. It keeps the completed combined 3B gold workflow's generator revision, gold tasks, generator and repair prompts, compact evidence, bounded recovery, declared-schema guard, sandbox and budgets. The repair revision is `a09a35458c702b33eeacc393d103063234e8bc28`.

The parser is bypassed. No adapters, retraining, GitHub push or previous prediction files are needed. One model is resident at a time. With T4 x2, generation uses GPU 1 and repair uses GPU 0. Runtime checks verify unquantized FP16 generation, actual NF4 repair parameters and float16 computation. The strict collector checks those recorded precision properties again.

## Run on Kaggle

1. Create a private dataset from `archive/stage2_combined_gold_7b_not_run/notebooks/stage2_combined_gold_7b/stage2_combined_gold_7b_source.zip`.
2. Import `archive/stage2_combined_gold_7b_not_run/notebooks/stage2_combined_gold_7b/stage2_combined_gold_7b_kaggle.ipynb` and attach that source dataset. Both an intact ZIP and Kaggle's automatically extracted form are supported.
3. Enable **Internet** and select **T4 x2**, matching the completed control's environment.
4. Use **Save Version → Run All**. The stages are explicitly labeled **2 smoke tasks → 20 development tasks → 100 full diagnostic evaluation tasks**. The 20-task stage is not the final evaluation. Each stage generates fresh outputs in its own directory.
5. Download **`stage2_combined_gold_7b.zip`**, **`stage2_combined_gold_7b_summary.zip`**, and the executed notebook. These are in `/kaggle/working/run_<timestamp>/`. The smoke and development bundles have `_smoke` and `_dev` suffixes; send the full unsuffixed 100-task bundle for analysis.

Preflight checks the source pack, all 100 test gold contracts after JSON serialization, and a scripted execution of the actual combined repair/guard. Smoke outcomes may be ACCEPT, REJECT or ESCALATE; infrastructure completion is separate from accuracy. Every final candidate is still checked by the original verifier plus schema guard. Failed inference exports a partial ZIP and log, which are not treated as complete results. CPU validation cannot guarantee GPU compatibility or improvement in model quality.

## Optional automatic comparison

Attach the completed control as a separate dataset:

`kaggle_outputs/stage2/combined_gold_v1/stage2_combined_gold_3b.zip`

It is used only by the optional last CPU cell, after the new inference and summary finish. No control outputs are reused during generation or repair. The cell supports intact ZIPs and automatically extracted sealed folders.

The comparison writes **`stage2_combined_gold_7b_comparison.zip`** and reports correct acceptance, incorrect acceptance, withholding, tokens, task IDs gained/lost and initial-code agreement. Control and new runs must match inputs, tasks, non-repair models, combined procedure, prompts, budgets, split and environment. Source versions and the intended repair model/precision policy differ explicitly. Any other mismatch stops comparison rather than silently treating unmatched runs as controlled evidence; the completed new inference ZIP remains available.

The control's collector JSON-key defect is handled by the packaged compatibility shim. In this new 7B collector, gold authenticity is compared against the exact JSON representation of each task's gold contract. Numeric mapping keys are handled correctly, while seals and all remaining content/settings checks remain enforced. The old sealed source, executed notebook and uploaded control ZIP are unchanged.

## How to interpret the result

The measured control is **84 correct, zero incorrect accepted and 16 withheld** with gold contracts and combined 3B FP16 repair. The new run tests whether the 7B NF4 repair configuration improves that result under the same combined procedure and budgets. Both size and precision change, so this is not an isolated model-size experiment.

The generator is run freshly with the same configuration; recorded initial-code agreement must be inspected before attributing individual gains solely to repair. The historical 94/100 gold result uses another repair protocol and is separate context. Gold contracts are an oracle diagnostic, and previously inspected tasks are not a fresh independent benchmark. Zero parser cost in these runs is not a deployment efficiency claim.

## Files and verification

- `archive/stage2_combined_gold_7b_not_run/eval/stage2_combined_gold_7b/`: new config, role-specific backend/precision checks, runner, strict collector and preflight. The execution boundary is byte-identical to the completed combined 3B gold version and calls the same frozen combined repair implementation.
- `archive/stage2_combined_gold_7b_not_run/notebooks/stage2_combined_gold_7b/make_notebook.py`: builder; embeds the exact old sealed source plus the separately versioned new modules.
- `archive/stage2_combined_gold_7b_not_run/tests/test_stage2_combined_gold_7b.py`: configuration isolation, actual precision validation, fresh scripted pipeline, control comparison and source/notebook loading.
- `tests/test_stage2_gold_collector_compat.py`: regression check for numeric JSON keys and rejection of an actually changed contract.
- `archive/stage2_combined_gold_7b_not_run/docs/stage2_combined_gold_7b_validation.json`: CPU validation record. GPU quality remains pending.
