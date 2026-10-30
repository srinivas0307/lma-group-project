# Fine-tuning the intent parser (Phase 7)

```
data (friend)                         training (Kaggle GPU)                    scoring (laptop)
finetune/datasets/*.csv               notebooks/kaggle/finetune_qlora_kaggle.ipynb    eval/score_parsers.py
finetune/data_gen/patterns + script     finetune/train_qlora.py  -> adapter      contract metrics + verdicts
check_examples.py -> checked.jsonl      eval/parse_contracts.py --adapter        + Stage 1 table
build_sft.py -> sft_train/val.jsonl       -> parse_ft_<model>/ (system.txt)
```

1. **Data:** follow `docs/finetune_data_guide.md`. Result: `finetune/data/sft_train.jsonl`, `sft_val.jsonl`.
2. **Smoke run on Kaggle:** `notebooks/kaggle/finetune_qlora_kaggle.ipynb` with `SMOKE = True` (50 steps per model).
   Check that memory fits and the loss falls (`ft_<model>_smoke.zip` → `train_log.jsonl`, `run_info.json`).
3. **Full run:** `SMOKE = False`. Trains 1.5B and 3B, then parses all evaluation tasks with each adapter
   (zero-shot, the format it was trained on). Unzip `parse_ft_qwen25_1_5b.zip` and `parse_ft_qwen25_3b.zip`
   to `kaggle_outputs/parsing/ft_1_5b/` and `kaggle_outputs/parsing/ft_3b/`; keep the `ft_*.zip` (adapters) as artifacts.
   Results land in `results/stage1/finetuned/intent_ft_*/`.
4. **Score:** `.venv/bin/python eval/score_parsers.py` → `intent_ft_1_5b`, `intent_ft_3b` next to the prompted
   parsers. **Gate 3:** does a fine-tuned parser match or beat `intent_prompted_7b` on contract F1?

Settings (`finetune/configs/qwen25_*.yaml`): QLoRA nf4 base, fp16 compute (T4), LoRA r=16 / alpha=32 / dropout 0.05
on all attention and MLP projections, lr 2e-4 cosine, 2 epochs, effective batch 16, max 2,048 tokens, loss on the
contract tokens only, seed 1234. Same settings for both sizes. Record any change here and in the run's `run_info.json`.

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
