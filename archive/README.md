# Archive index

These artifacts preserve the project's research history. The current final evaluation is documented in [the six-notebook guide](../docs/stage2_six_notebooks.md). Start from the [project README](../README.md) for active work.

## Notebooks

- [Earlier core templates](notebooks/core/) — candidate generation, parsing, agent evaluation, Stage 2 development, and the superseded combined/resume evaluation. The recovery builder remains beside its input notebook so historical regression tests and regeneration still work.
- [Optional extension templates](notebooks/extensions/) — re-ask, pre-check combinations, tool fine-tuning, LLM verifier, vLLM, Gradio, and agent extensions. These are outside the current required six-system evaluation.
- [Executed notebooks](notebooks/executed/) — grouped into `stage1/`, `training/`, `agent/`, `stage2_dev/`, and `stage2_test/`. Notebook contents and outputs are preserved.
- [Extracted notebook pack](distributions/stage2_six_notebooks/) — a previously extracted copy; use the canonical current [ZIP](../notebooks/kaggle/stage2_six_notebooks.zip).

The active parser training template is in [notebooks/training/](../notebooks/training/).

## Stage 2 history

- [Downloaded outputs](stage2/downloaded_outputs/) — original development bundles and the failed GPU-split test attempt.
- [Root checkpoint](stage2/test/root_checkpoint/) — the earlier root-level `stage2_test_bundle.zip`, preserved separately from other checkpoints.
- [Early evaluation](stage2/early_evaluation/) — earlier combined Stage 2 output, retained as historical evidence.
- [Prompt development](stage2/prompt_development/) — Phase A/B scored runs supporting the selected prompts. The [development report](../results/STAGE2_DEV_RESULTS.md) links here.

Do not feed these historical bundles to the new collector. They belong to earlier experiment/runtime versions.

## Optional experiment artifacts and reviews

- [Extension results](extensions/results/) — no-progress replay and Docker/sandbox comparisons. Stage 1 extension outputs remain in `results/stage1/extensions/` because they share the reproduction tree and existing documentation.
- [Fine-tuning review material](review/finetune_data/) — review scripts, decisions, original review sheets, and screenshots previously under `tmp/review/`.
- The exported mid-submission PDF is now in [docs/reports/](../docs/reports/).

## Existing historical collections

The earlier `pools/`, `generation/`, `results/`, `configs/`, `agent_scenarios/`, and legacy files directly under `notebooks/` remain at their original archive paths. Some are exercised by regression tests; moving or deleting them would weaken reproducibility.

Historical plans and workflow instructions are grouped under [docs/history/](../docs/history/): the full roadmap, extension plan/registry, Phase 1 brief, Stage 2 development guide, and combined evaluation/recovery guide. Earlier instructions describe their experiment's assumptions and may require older environments.

## Move manifest

[cleanup_manifest_2026-11-13.json](cleanup_manifest_2026-11-13.json) records each old and new location, file size, and SHA-256 before and after the cleanup. Files were checksum-verified immediately after moving. Markdown link updates are identified by `content_changed`; notebooks, raw runs, checkpoints, and review evidence retain their contents.

No experiment data or model weights were deleted. The trained adapters remain under `finetune/runs/`; generation/parser inputs, frozen source, benchmark data, and the local environment retain their active paths.
