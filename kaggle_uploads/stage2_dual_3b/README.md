# Generated ZIPs removed during report cleanup

The large adapter-containing upload ZIPs are rebuildable copies. Canonical source ZIPs and original training outputs remain in the repository. To recreate these packages from the repository root:

```bash
.venv/bin/python kaggle_uploads/stage2_dual_3b/make_kaggle_datasets.py
```

The current repair-development notebook uses only its own small source ZIP and requires no adapter dataset. See `docs/stage2_repair_dev.md`.

# Kaggle inputs: unquantized 3B generator and repair

Use stage2_dual_3b_selective_kaggle_dataset.zip for the selective-facts notebook, and stage2_dual_3b_tools_kaggle_dataset.zip for the tool-use notebook. Each is a clean, single-upload dataset containing the matching source and selected adapter. Neither contains nested archives or checkpoint results.

Both generation and repair use the same Qwen2.5-3B-Instruct revision with full FP16 weights. Intent parsers keep their existing NF4 loading. Instructions: docs/stage2_dual_3b.md.
