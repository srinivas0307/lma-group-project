# Generated ZIPs removed during report cleanup

The large adapter-containing upload ZIPs are rebuildable copies. Canonical source ZIPs and original training outputs remain in the repository. To recreate these packages from the repository root:

```bash
.venv/bin/python archive/stage2_dual_7b/kaggle_uploads/stage2_dual/make_tools_dataset.py
```

The current repair-development notebook uses only its own small source ZIP and requires no adapter dataset. See `docs/stage2_repair_dev.md`.

# Clean Kaggle tool-use Stage 2 inputs

Upload **stage2_dual_tools_kaggle_dataset.zip** as one private Kaggle dataset and attach it to the tools proposed notebook. It contains the matching source and exact selected adapter, without nested ZIPs or training-result folders. This avoids Kaggle's baseline/epoch directory collisions.

Alternative: use tools_adapter/tool_use_train_1_5b.zip alongside an already attached source dataset. This is adapter-only, unlike the full training-output archive of the same basename.

Original model weights, metadata, results and inference/runtime source are unchanged. Full instructions: archive/stage2_dual_7b/docs/stage2_dual.md.
