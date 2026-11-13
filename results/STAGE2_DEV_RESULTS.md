# Stage 2 development runs: artifact index

Imported and organized on 2026-11-12. Raw downloads, saved notebook outputs and result records are preserved without editing their contents. These are development runs, not held-out test results.

## Original Stage 2 development run

- Source commit: `8ea400a7c3b8c9937c44e1b6cb99383ce2a74474`.
- Systems: direct baseline and gold-contract pipeline, using the original b2-v1 / a1-v4 configuration where applicable.
- Raw downloaded folder: [archive/stage2/downloaded_outputs/dev/original](../archive/stage2/downloaded_outputs/dev/original).
- Usable result folder: [results/stage2](../archive/stage2/early_evaluation).
- Summary: [stage2_metrics_dev.json](../archive/stage2/early_evaluation/stage2_metrics_dev.json).

## Prompt comparison: Phase A

- Executed notebook: [stage2_prompt_dev_phase_a.ipynb](../archive/notebooks/executed/stage2_dev/stage2_prompt_dev_phase_a.ipynb), formerly `stage-2_updated.ipynb`.
- Raw bundle: [prompt_phase_a.zip](../archive/stage2/downloaded_outputs/dev/prompt_phase_a.zip), formerly `stage2_prompt_dev_bundle.zip`.
- Results: [stage2_prompt_dev/phase_a](../archive/stage2/prompt_development/phase_a).
- Summary: [stage2_metrics_dev.json](../archive/stage2/prompt_development/phase_a/stage2_metrics_dev.json).
- Three arms: direct baseline, gold b2-v1 / a1-v4, and gold b2-v2 / a1-v4. Each has 20 development task/seed records.

## Prompt comparison: Phase B

- Executed notebook: [stage2_prompt_dev_phase_b.ipynb](../archive/notebooks/executed/stage2_dev/stage2_prompt_dev_phase_b.ipynb), formerly `stage-2(1).ipynb`.
- Raw bundle: [prompt_phase_b.zip](../archive/stage2/downloaded_outputs/dev/prompt_phase_b.zip), formerly `stage2_prompt_dev_bundle(1).zip`.
- Results: [stage2_prompt_dev/phase_b](../archive/stage2/prompt_development/phase_b).
- Summary: [stage2_metrics_dev.json](../archive/stage2/prompt_development/phase_b/stage2_metrics_dev.json).
- Four arms: the three Phase A configurations plus gold b2-v2 / a1-v5. Each has 20 development task/seed records.

Both prompt-comparison bundles record source commit `e9aaa31b85225fc14792c3d34b9f24de149f92b1` and independently verified sandbox version sb1. Phase B contains its own control-run snapshots; they are not merged with or substituted for Phase A records. Historical Kaggle paths and bundle names inside metadata and notebook output remain unchanged as provenance.

The development selection is **b2-v2 / a1-v4 / sb1**. Both a1-v4 and a1-v5 finished 19/20 tasks correctly with b2-v2; a1-v5 did not improve correctness and used more tokens. These oracle-contract results do not establish final accuracy with learned contracts. See the [six-notebook test workflow](../docs/stage2_six_notebooks.md) for the next experiment.

## Integrity and scoring

The [artifact manifest](../archive/stage2/downloaded_outputs/dev/artifact_manifest.json) records the original-to-current mapping and SHA-256 hashes for 17 moved original files and 63 copied/extracted result files. The raw archives were renamed, not recompressed. The original `stage2_bundle_dev/` wrapper is now under `archive/stage2/downloaded_outputs/dev/original/`.

For Phase B, recompute metrics with:

```bash
.venv/bin/python eval/stage2_metrics.py archive/stage2/prompt_development/phase_b dev
```

Use `phase_a` for Phase A or `archive/stage2/early_evaluation` for the original run. Do not score the parent `archive/stage2/prompt_development/` directory: its immediate children are snapshots, not systems. Re-scoring writes derived metrics; the raw bundles remain the unchanged source copies.
