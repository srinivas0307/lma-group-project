"""QLoRA fine-tuning of the intent parser (README Phase 7). Runs on Kaggle (GPU); a CPU smoke run is possible.

    python finetune/train_qlora.py --config finetune/configs/qwen25_3b.yaml
    python finetune/train_qlora.py --config finetune/configs/smoke_cpu.yaml        # tiny CPU check

Input: chat-format SFT files from finetune/data_gen/build_sft.py ({"messages": [system, user, assistant]}).
Loss is computed on the assistant tokens only (the contract), not on the prompt.
Examples longer than max_seq_len are dropped (reported), never truncated, so no contract is cut in half.
Plain transformers Trainer + peft (no trl), so the training loop is visible and version-stable.

Outputs in <output_dir>/:
    adapter/          LoRA adapter + tokenizer (load with eval/parse_contracts.py --adapter)
    train_log.jsonl   loss / eval loss per logging step
    run_info.json     config, library versions, data hashes, example counts, runtime
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import yaml

# Less fragmentation on the 16 GB T4 (the OOM message suggests it); set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
# Running `python finetune/train_qlora.py` puts finetune/ on sys.path, where finetune/datasets/ (the training
# CSVs) would shadow the HuggingFace `datasets` library that transformers imports. Remove it.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

DEFAULTS = dict(
    base_model="Qwen/Qwen2.5-3B-Instruct", train_file="finetune/data/sft_train.jsonl",
    val_file="finetune/data/sft_val.jsonl", output_dir="finetune/runs/qwen25_3b",
    load_in_4bit=True, compute_dtype="float16", gradient_checkpointing=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    learning_rate=2e-4, num_epochs=2, max_steps=-1, warmup_ratio=0.03, weight_decay=0.0, lr_scheduler="cosine",
    per_device_batch_size=4, gradient_accumulation_steps=4, max_seq_len=2048,
    logging_steps=10, eval_steps=100, save_steps=0, seed=1234, limit_train=None, limit_val=None,
    optim=None,
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path, limit, sample_seed=None):
    """limit: keep `limit` rows; the first ones, or a seeded random sample if sample_seed is given
    (the validation file is ordered by type, so its first rows would cover only one or two types)."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        if roles[-1] != "assistant" or roles[0] != "system":
            raise ValueError(f"{path}: example {r.get('id')} must be [system, user, ..., assistant]")
    if not limit or limit >= len(rows):
        return rows
    if sample_seed is None:
        return rows[:limit]
    keep = sorted(random.Random(sample_seed).sample(range(len(rows)), limit))
    return [rows[i] for i in keep]


def tokenize(rows, tok, max_len):
    """input_ids + labels with the prompt masked (-100). Returns (examples, n_dropped_too_long)."""
    out, dropped = [], 0
    for r in rows:
        msgs = r["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise ValueError("chat template: the full conversation doesn't start with the prompt")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        if f_ids[:len(p_ids)] != p_ids:            # tokenisation boundary moved; tokenise the answer separately
            f_ids = p_ids + tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        if len(f_ids) > max_len:
            dropped += 1
            continue
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        out.append(dict(input_ids=f_ids, labels=labels, attention_mask=[1] * len(f_ids)))
    return out, dropped


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]       # right padding for training
        return dict(input_ids=torch.tensor(pad([b["input_ids"] for b in batch], self.pad_id)),
                    labels=torch.tensor(pad([b["labels"] for b in batch], -100)),
                    attention_mask=torch.tensor(pad([b["attention_mask"] for b in batch], 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config values, e.g. max_steps=50")
    args = ap.parse_args()
    cfg = dict(DEFAULTS, **(yaml.safe_load(args.config.read_text()) or {}))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        sys.exit(f"unknown config keys: {sorted(unknown)}")

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    out = (ROOT / cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    if cfg["load_in_4bit"] and not cuda:
        sys.exit("load_in_4bit needs a CUDA GPU; set load_in_4bit: false for CPU runs")
    dtype = getattr(torch, cfg["compute_dtype"])

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_rows = load_rows(ROOT / cfg["train_file"], cfg["limit_train"])
    val_rows = load_rows(ROOT / cfg["val_file"], cfg["limit_val"], sample_seed=cfg["seed"]) if cfg["val_file"] else []
    train_ds, drop_tr = tokenize(train_rows, tok, cfg["max_seq_len"])
    val_ds, drop_va = tokenize(val_rows, tok, cfg["max_seq_len"])
    if not train_ds:
        sys.exit("no training examples left after tokenisation")
    lens = sorted(len(x["input_ids"]) for x in train_ds)
    for k in ("train_file", "val_file"):                      # which data this run really uses
        if cfg[k]:
            print(f"{k}: {cfg[k]} sha256 {sha256(ROOT / cfg[k])[:12]}", flush=True)
    print(f"train {len(train_ds)} (dropped {drop_tr} too long) | val {len(val_ds)} (dropped {drop_va}) | "
          f"tokens median {lens[len(lens) // 2]} max {lens[-1]}", flush=True)

    kw = dict(dtype=dtype if cuda else torch.float32)
    if cfg["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kw)
    model.config.use_cache = False
    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
    elif cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                             lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"],
                                             bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)", flush=True)

    log_path = out / "train_log.jsonl"
    log_path.write_text("")

    class Log(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **k):
            if logs:
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(dict(step=state.global_step, **logs)) + "\n")

    steps_per_epoch = math.ceil(len(train_ds) / (cfg["per_device_batch_size"] * cfg["gradient_accumulation_steps"]))
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"], learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_epochs"], max_steps=cfg["max_steps"], warmup_steps=max(1, int(
            cfg["warmup_ratio"] * (cfg["max_steps"] if cfg["max_steps"] > 0 else steps_per_epoch * cfg["num_epochs"]))),
        weight_decay=cfg["weight_decay"], lr_scheduler_type=cfg["lr_scheduler"], logging_steps=cfg["logging_steps"],
        eval_strategy="steps" if val_ds else "no", eval_steps=cfg["eval_steps"],
        save_strategy="steps" if cfg["save_steps"] else "no", save_steps=cfg["save_steps"] or 500,
        fp16=cuda and cfg["compute_dtype"] == "float16", bf16=cuda and cfg["compute_dtype"] == "bfloat16",
        optim=cfg["optim"] or ("paged_adamw_8bit" if cfg["load_in_4bit"] else "adamw_torch"),
        gradient_checkpointing=False, report_to=[], seed=cfg["seed"], data_seed=cfg["seed"],
        remove_unused_columns=False, dataloader_num_workers=0, use_cpu=not cuda,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds or None,
                      data_collator=Collator(tok.pad_token_id), callbacks=[Log()])
    t0 = time.time()
    result = trainer.train()
    secs = time.time() - t0
    final_eval = trainer.evaluate() if val_ds else {}

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    versions = sorted({r.get("prompt_version", "unknown") for r in train_rows})
    if len(versions) != 1:
        sys.exit(f"training data mixes parser prompt versions {versions}; rebuild it with build_sft.py")
    (out / "adapter" / "igt_prompt_version.txt").write_text(versions[0])      # parse_contracts.py reads this
    logs = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    losses = [l["loss"] for l in logs if "loss" in l]
    info = dict(config=cfg, base_model=cfg["base_model"], parser_prompt_version=versions[0], runtime_s=round(secs, 1),
                train_loss=result.training_loss, final_eval=final_eval,
                first_logged_loss=losses[0] if losses else None, last_logged_loss=losses[-1] if losses else None,
                n_train=len(train_ds), n_val=len(val_ds), dropped_too_long=dict(train=drop_tr, val=drop_va),
                max_tokens=lens[-1], steps=result.global_step, trainable_params=trainable,
                data_sha256={k: sha256(ROOT / cfg[k]) for k in ("train_file", "val_file") if cfg[k]},
                versions=dict(python=platform.python_version(), torch=torch.__version__,
                              transformers=transformers.__version__,
                              peft=__import__("peft").__version__),
                gpu=torch.cuda.get_device_name(0) if cuda else None)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps({k: info[k] for k in ("runtime_s", "steps", "train_loss", "first_logged_loss",
                                            "last_logged_loss", "final_eval")}, indent=2, default=str))
    print(f"adapter saved to {out / 'adapter'}")


if __name__ == "__main__":
    main()
