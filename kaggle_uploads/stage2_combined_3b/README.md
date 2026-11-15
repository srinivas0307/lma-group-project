# Optional combined-workflow upload packages

Prefer attaching the new small source ZIP and your existing selected adapter dataset from the completed 3B run. No new adapter weights are needed.

If you need a new single-upload dataset for each method, run from the repository root:

```bash
.venv/bin/python kaggle_uploads/stage2_combined_3b/make_kaggle_datasets.py
```

This builds `stage2_combined_3b_selective_kaggle_dataset.zip` and `stage2_combined_3b_tools_kaggle_dataset.zip` from the canonical training outputs and new source package. It includes only the selected adapter files, with verified hashes; no nested training/checkpoint ZIPs. Large generated copies are not stored by default. See [the workflow guide](../../docs/stage2_combined_3b.md).
