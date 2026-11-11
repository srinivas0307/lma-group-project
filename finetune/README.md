# Fine-tuning the intent parser (Phase 7)

**Full runs imported and scored (2026-11-11):** both 1.5B and 3B adapters completed two epochs and parsed all 120 evaluation tasks. See [the analysis and results](../results/FINETUNE_RESULTS.md) and [the executed notebook](../notebooks/executed/finetune_qlora_full.ipynb).

Adapters and original training artifacts are in `finetune/runs/qwen25_1_5b/` and `finetune/runs/qwen25_3b/` (git-ignored). Parser outputs are in `kaggle_outputs/parsing/ft_1_5b/` and `ft_3b/`. Scored results and copies of portable training metadata are in `results/stage1/finetuned/intent_ft_1_5b/` and `intent_ft_3b/`.

```
data (laptop)                           training (Kaggle GPU)                           scoring (laptop)
prepare_datasets.py -> datasets/*.csv   notebooks/kaggle/finetune_qlora_kaggle.ipynb    eval/score_parsers.py
generate_examples.py -> raw/*.jsonl     finetune/train_qlora.py  -> adapter             contract metrics + verdicts
check_examples.py -> checked.jsonl      eval/parse_contracts.py --adapter               + Stage 1 table
build_sft.py -> sft_train/val.jsonl       -> parse_ft_<model>/ (system.txt)
review_sample.py -> review.csv (hand review)
```

1. **Data** (`docs/finetune_data_guide.md`; all CPU, about 20 minutes):
   ```
   .venv/bin/python finetune/data_gen/prepare_datasets.py     # 8 snapshots from data/finetune/ -> finetune/datasets/ (+ README with sha256)
   .venv/bin/python finetune/data_gen/generate_examples.py    # 40 patterns -> finetune/data/raw/generated.jsonl
   .venv/bin/python finetune/data_gen/check_examples.py finetune/data/raw/generated.jsonl --out finetune/data/checked.jsonl --workers 10
   .venv/bin/python finetune/data_gen/build_sft.py finetune/data/checked.jsonl --val-datasets student
   .venv/bin/python finetune/data_gen/review_sample.py        # 100 examples to hand-check -> finetune/data/review.csv
   ```
   Result: `finetune/data/sft_train.jsonl`, `sft_val.jsonl`. The `student` dataset is validation only.
2. **Smoke run on Kaggle:** `notebooks/kaggle/finetune_qlora_kaggle.ipynb` with `SMOKE = True` (50 steps per model).
   Check that memory fits and the loss falls (`ft_<model>_smoke.zip` → `train_log.jsonl`, `run_info.json`).
3. **Full run:** `SMOKE = False`. Trains 1.5B and 3B, then parses all evaluation tasks with each adapter
   (zero-shot, the format it was trained on). Unzip `parse_ft_qwen25_1_5b.zip` and `parse_ft_qwen25_3b.zip`
   to `kaggle_outputs/parsing/ft_1_5b/` and `kaggle_outputs/parsing/ft_3b/`; keep the `ft_*.zip` (adapters) as artifacts.
   Results land in `results/stage1/finetuned/intent_ft_*/`.
4. **Score:** `.venv/bin/python eval/score_parsers.py` → `intent_ft_1_5b`, `intent_ft_3b` next to the prompted
   parsers. **Gate 3:** does a fine-tuned parser match or beat `intent_prompted_7b` on contract F1?

Shared settings: QLoRA nf4 base, fp16 compute (T4), LoRA r=16 / alpha=32 / dropout 0.05 on attention and MLP projections, lr 2e-4 cosine, per-device batch 1, gradient accumulation 16, max 3,072 tokens, assistant-contract loss only, seed 1234. The current `finetune/configs/qwen25_*.yaml` files specify one epoch; the imported full runs used **two epochs**, as recorded in their `run_info.json`. Runtime device count/global batch were not recorded, so do not describe the global effective batch as verified. The Kaggle notebook used the corrected `finetune/data/v2/` files; direct training with the YAML defaults uses `finetune/data/` unless overridden.

**Local CPU check** (pipeline only, no quality): a separate environment with torch, transformers and peft, then
```
python finetune/data_gen/smoke_examples.py
python finetune/data_gen/check_examples.py finetune/data/smoke/raw.jsonl --out finetune/data/smoke/checked.jsonl
python finetune/data_gen/build_sft.py finetune/data/smoke/checked.jsonl --val-datasets smoke_shop_val --out-dir finetune/data/smoke
python finetune/train_qlora.py --config finetune/configs/smoke_cpu.yaml        # Qwen2.5-0.5B, 6 steps, ~3 min
python eval/parse_contracts.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter finetune/runs/smoke_cpu/adapter \
    --dtype float32 --limit 2 --batch 1 --out /tmp/ftparse
```
Checked on 2026-10-29: loss 0.55 → 0.05 in 6 steps, eval loss 0.18 → 0.07, adapter reloads and gives identical
output twice, output parses as a valid contract. The smoke data is synthetic pipeline-test data, **not** training data.
