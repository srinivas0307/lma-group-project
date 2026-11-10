"""QLoRA training for the tool-using fine-tuned parsers (extension #13, ft-t1). Same script as finetune/train_qlora.py
(imported, not copied, and not edited) with one change: in multi-turn examples the loss covers EVERY assistant turn
(the tool call and the final contract), not only the last message. Tool results and prompts stay masked.
Single-turn examples are tokenised by train_qlora.tokenize itself, so they are identical to the current training
(tests/extensions/test_ft_tools.py checks this with a stub tokenizer).

    python finetune/train_qlora_tools.py --config finetune/configs/qwen25_1_5b_tools.yaml
    python finetune/train_qlora_tools.py --config finetune/configs/qwen25_3b_tools.yaml --set max_steps=20   # smoke
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_train_qlora():
    spec = importlib.util.spec_from_file_location("train_qlora", HERE / "train_qlora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # its module code also removes finetune/ from sys.path (datasets shadowing)
    return mod


tq = _load_train_qlora()
_single_turn_tokenize = tq.tokenize


def tokenize_multi_turn(rows, tok, max_len):
    """Like train_qlora.tokenize, but every assistant turn is trained. Returns (examples, n_dropped_too_long)."""
    out, dropped = [], 0
    for r in rows:
        msgs = r["messages"]
        if sum(m["role"] == "assistant" for m in msgs) <= 1:
            ex, d = _single_turn_tokenize([r], tok, max_len)
            out += ex
            dropped += d
            continue
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        spans, pos = [], 0                  # (text, trained?) pieces that concatenate to `full`
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            start = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
            end = tok.apply_chat_template(msgs[:i + 1], tokenize=False, add_generation_prompt=False)
            if not (full.startswith(start) and full.startswith(end) and len(start) >= pos):
                raise ValueError("chat template: a conversation prefix is not a prefix of the full conversation")
            spans += [(full[pos:len(start)], False), (full[len(start):len(end)], True)]
            pos = len(end)
        spans.append((full[pos:], False))
        ids, labels = [], []
        for text, trained in spans:
            if not text:
                continue
            t = tok(text, add_special_tokens=False)["input_ids"]
            ids += t
            labels += t if trained else [-100] * len(t)
        if len(ids) > max_len:
            dropped += 1
            continue
        out.append(dict(input_ids=ids, labels=labels, attention_mask=[1] * len(ids)))
    return out, dropped


def main():
    tq.tokenize = tokenize_multi_turn       # train_qlora.main looks the name up at call time
    tq.main()


if __name__ == "__main__":
    main()
