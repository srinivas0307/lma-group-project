# Downloaded Stage 2 artifacts

`dev/` preserves the uploaded development downloads:

- `original/`: the original extracted `stage2_bundle_dev/` folder, including its `results/stage2/` archive layout.
- `prompt_phase_a.zip`: generator-prompt comparison, originally `stage2_prompt_dev_bundle.zip`.
- `prompt_phase_b.zip`: repair-prompt comparison plus control runs, originally `stage2_prompt_dev_bundle(1).zip`.
- `artifact_manifest.json`: moved-file mappings and checksums, plus source mappings for canonical result files.

Executed notebooks are in `archive/notebooks/executed/stage2_dev/stage2_prompt_dev_phase_a.ipynb` and `stage2_prompt_dev_phase_b.ipynb`. These are distinct from the unexecuted templates in `notebooks/kaggle/`.

Usable results are in `results/stage2/`, `results/stage2_prompt_dev/phase_a/` and `results/stage2_prompt_dev/phase_b/`. Keep snapshots separate when extracting or scoring. See [the artifact index](../../../results/STAGE2_DEV_RESULTS.md) for links, experiment identities and commands.
