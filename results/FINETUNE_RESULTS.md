# Full QLoRA fine-tuning results and analysis

Analysis and artifact import: **2026-11-11**. These results come from the uploaded full Kaggle run, not the earlier smoke notebook.

Both Qwen2.5 adapters completed training and generated contracts for all 120 evaluation tasks. Fine-tuning substantially improves contract validity and Stage 1 verification compared with the prompted 7B baseline. It does not yet establish end-to-end Stage 2 generation-and-repair performance.

## Artifact locations

- Executed full-run notebook: [finetune_qlora_full.ipynb](../archive/notebooks/executed/training/finetune_qlora_full.ipynb).
- Training logs, adapters, and tokenizers: `finetune/runs/qwen25_1_5b/` and `finetune/runs/qwen25_3b/`.
- Downloaded parser outputs: [ft_1_5b](../kaggle_outputs/parsing/ft_1_5b) and [ft_3b](../kaggle_outputs/parsing/ft_3b).
- Contract metrics, full/change-only/preserve-only verdicts, and portable training metadata: [intent_ft_1_5b](stage1/finetuned/intent_ft_1_5b) and [intent_ft_3b](stage1/finetuned/intent_ft_3b).
- Shared comparisons: [contract_summary.json](stage1/summary/contract_summary.json) and [metrics_stage1.json](stage1/summary/metrics_stage1.json).
- Original-to-current path mapping and SHA-256 checksums: [artifact_manifest.json](stage1/finetuned/artifact_manifest.json).

The original `finetune_outputs/` directories were moved into the locations above, and the root notebook was moved into `archive/notebooks/executed/`. All 29 supplied files were verified byte-for-byte after moving. Small training metadata files are also copied under each results folder's `training/` directory so they remain available independently of the large adapters. `finetune/runs/` is intentionally git-ignored; moving adapters there does not add them to version control.

## Training evidence

Both runs record source commit `8e42ab7607a6aa4abd0f1e8b09ccd3d47c2ecc48`, seed 1234, prompt `p1-v2`, four-bit NF4 loading, FP16 compute, LoRA rank 16 / alpha 32 / dropout 0.05, learning rate `2e-4`, and a cosine schedule. Attention and MLP projections are adapted. Training loss is computed on assistant contract tokens.

| Measure | Qwen2.5-1.5B | Qwen2.5-3B |
|---|---:|---:|
| Training examples | 3,614 | 3,614 |
| Validation examples monitored | 200 of 796 | 200 of 796 |
| Completed epochs | 2 | 2 |
| Recorded optimizer steps | 226 | 226 |
| Training runtime, excluding subsequent parsing | 4.04 h | 7.15 h |
| First logged training loss | 0.442310 | 0.453723 |
| Last logged training loss | 0.002819 | 0.002042 |
| Final validation loss | 0.013758 | 0.008077 |
| Adapter parameters | 18,464,768 | 29,933,568 |
| Examples dropped for sequence length | 0 | 0 |

The maximum recorded training length is 1,827 tokens, below the 3,072-token limit. Both adapters have consistent tensor counts, parameter totals, and safetensors payload lengths. This is a structural integrity check; the analysis did not reload models locally on a GPU. The notebook's subsequent inference demonstrates that the saved adapters were loaded in the original run.

The dataset hashes match the local revised v2 files exactly:

- Training: `4222330ebad0e91b5497c64d17721e01a4f66bad90c6a834a25553eea8b46597`.
- Validation: `79e9db54a13fda41d2fda42c8b4b17a2d4edb062d6601e9c7f8717a3d922c9e3`.

The notebook copies v2 files into `finetune/data/` before training, explaining the non-versioned paths in `run_info.json`. Monitoring uses the same fixed random 200-example subset of the held-out Student dataset for both models. It does not use the 120 benchmark tasks.

Validation loss decreases from 0.017676 to 0.013758 for 1.5B. For 3B it decreases from 0.013886 at step 100 to 0.007926 at step 200, then rises slightly to 0.008077 at completion. That small rise alone does not establish serious overfitting. Lower token loss does not necessarily mean better semantic understanding of requests.

## Contract generation: 100 test tasks

Scores use `eval/contract_metrics.py`, with invalid outputs treated as empty predictions and micro F1 over normalized predicate keys.

| Parser | Invalid contracts | Must-change F1 | Must-preserve F1 | Exact match |
|---|---:|---:|---:|---:|
| Prompted 7B, p1-v3 | 13% | 0.751 | 0.906 | 27% |
| Fine-tuned 1.5B, p1-v2 | 2% | 0.836 | 0.972 | 56% |
| Fine-tuned 3B, p1-v2 | 2% | 0.788 | 0.977 | 56% |

Both fine-tuned models exceed the prompted 7B baseline on both section F1 scores. Fine-tuned inference is zero-shot; the prompted baseline uses its dev-selected few-shot p1-v3 prompt. This is the project's system comparison, not an experiment holding prompt and few-shot context identical.

All 120 tasks have one parse from each model. Across dev and test, 1.5B has 118 valid outputs, one invalid JSON output, and one schema-invalid contract. The 3B model has 116 valid outputs and four schema-invalid contracts. Neither run has token-truncated outputs.

| Task type, 20 test tasks each | 1.5B exact match | 3B exact match |
|---|---:|---:|
| Column transformation | 65% | 55% |
| Deduplication | 85% | 70% |
| Filtering | 60% | 80% |
| Missing values | 55% | 65% |
| Type conversion | 15% | 10% |

Exact match is structural rather than a proof of logical equivalence. Missing a redundant null-preservation check can lower exact match without changing decisions. Conversely, schema-valid contracts can contain substantively incorrect conditions, literals, or category orderings. High preservation F1 should not be presented as transformation accuracy.

## Stage 1 verification: frozen pool v4

The generated contracts were applied to the unchanged cached candidate outputs using `eval/run_intent_oracle.py`. Metrics use `eval/metrics.py`, including 2,000 task-level bootstrap resamples. No GPU inference, code generation, or repair was rerun for this scoring.

Test contains 671 candidates: 630 runnable, including 213 wrong outputs and 417 correct outputs. Recall excludes execution failures. Both rejection and escalation count as not accepted for recall and false rejection. Silent error rate means wrong among accepted outputs.

| System | Wrong-output recall | False rejection | Silent error rate | Escalation |
|---|---:|---:|---:|---:|
| Conventional checks | 45.5% | 13.9% | 24.4% | 0% |
| Prompted 7B | 89.2% | 25.4% | 6.9% | 14.8% |
| Fine-tuned 1.5B | 94.4% | 16.5% | 3.3% | 3.1% |
| Fine-tuned 3B | 91.5% | 15.8% | 4.9% | 3.1% |
| Gold-contract oracle | 94.4% | 0% | 2.8% | 0.6% |

The 1.5B model catches 201/213 runnable wrong candidates, accepts 348/417 correct candidates, and accepts 12 wrong candidates. Compared with prompted 7B, it catches 11 additional wrong outputs and accepts 37 additional correct outputs. Its recall matches the oracle numerically, but the missed-error sets differ: 11 misses are shared, and each has one additional distinct miss. Its false rejection remains above the oracle's.

The 3B model catches 195/213 runnable wrong candidates, accepts 351/417 correct candidates, and accepts 18 wrong candidates. It accepts three more correct candidates than 1.5B but also six more wrong candidates.

95% task-bootstrap intervals:

- 1.5B: recall 89.7–98.1%; false rejection 9.0–25.0%; silent error rate 1.2–5.9%.
- 3B: recall 84.9–96.8%; false rejection 8.9–24.0%; silent error rate 1.9–8.9%.

These are single-seed parser runs on a finite benchmark. Candidates from the same task are correlated, and gold-contract completeness and annotation limitations still apply.

## Remaining errors

- **Statistics are guessed.** For Titanic median-age filling, the actual median is 28, but 1.5B specifies 37 and 3B specifies 26. These parsers see a schema and a small row sample, not a tool-computed median.
- **Category order is misunderstood.** The requested Titanic Pclass order is `[3, 2, 1]`; 1.5B emits `[1, 3, 2]` with a mapping, while 3B emits `[1, 2, 3]`.
- **Deduplication keys can be wrong.** Some Airbnb contracts use all columns or incorrect exclusions instead of the requested key.
- **Conditions can target the wrong rows.** Missing-value tasks include incorrect conditions or reversed preservation conditions.
- **Formatting failures remain.** The 1.5B test failures are `adult_col_005` (invalid JSON) and `airbnb_miss_004` (unsupported `order` field). The 3B failures are `adult_dedup_006` and `airbnb_dedup_006` in dev, and `taxi_dedup_005` and `titanic_col_001` in test. These are not truncation failures.

## Model choice and notebook caveats

The 1.5B model leads on must-change F1, test wrong-output detection, and training cost. The 3B model has slightly higher preservation F1 and lower false rejection. Development criteria also differ: 1.5B has zero invalid dev contracts and mean section F1 0.936, versus two invalid dev contracts and mean F1 0.873 for 3B. However, dev candidate-pool false rejection is 19.4% for 1.5B and 6.0% for 3B. Declare the selection criterion before further test-driven changes; neither model is uniformly superior.

The uploaded run metadata records two epochs, whereas the current local YAML files specify one. Preserve the recorded run configuration when reporting these results. Both runs record 226 steps; the notebook comment estimates 452 from nominal single-device arithmetic. GPU count and actual global batch size were not recorded, so that discrepancy cannot be resolved conclusively from these artifacts. Report the observed steps and completed epochs without inferring a hardware configuration.

Checkpoint saving was disabled. The exported adapters are the final models; intermediate or best-validation adapters were not saved. The notebook completed despite Hugging Face authentication and gradient-checkpointing warnings.

The full training and parser-inference stage is complete. End-to-end generation and repair with these adapters remains a separate Stage 2 experiment.

## Reproduction

From the project root, the existing scoring workflow now discovers both imported parser folders:

```bash
.venv/bin/python eval/score_parsers.py
```

This recomputes all parser contract scores and Stage 1 comparisons, writing result files. To regenerate only one model's contract metrics:

```bash
.venv/bin/python eval/contract_metrics.py \
  --parses kaggle_outputs/parsing/ft_1_5b/parses.jsonl \
  --out results/stage1/finetuned/intent_ft_1_5b/contract_metrics.json
```

Use `finetune/runs/qwen25_1_5b/adapter` or `finetune/runs/qwen25_3b/adapter` as the adapter path for local model loading. Historical Kaggle paths inside the notebook and original metadata are preserved as provenance.
