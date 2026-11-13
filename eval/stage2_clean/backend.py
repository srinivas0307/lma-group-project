"""Explicit model placement, pinned revisions, real GPU probes and per-call memory diagnostics."""
import importlib.metadata
import platform
import time

from .attention import configure_attention
from .config import EXPECTED, MODELS


def environment():
    import torch
    packages = {}
    for name in (*EXPECTED, "accelerate", "tokenizers", "safetensors", "huggingface-hub"):
        packages[name] = importlib.metadata.version(name)
    mismatch = {k: (packages[k], expected) for k, expected in EXPECTED.items() if packages[k] != expected}
    if mismatch:
        raise RuntimeError(f"Inference package mismatch (actual, expected): {mismatch}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Select GPU T4 x2 in Kaggle settings")
    gpus = [torch.cuda.get_device_name(i) for i in range(2)]
    if any("T4" not in name for name in gpus):
        raise RuntimeError(f"This experiment requires T4 x2; found {gpus}")
    return dict(packages=packages, python=platform.python_version(), cuda=torch.version.cuda, gpus=gpus,
                capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(2)])


def preflight(record):
    import torch
    configure_attention(torch, record)


class Backend:
    def __init__(self, role, record):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch, self.record, self.role = torch, record, role
        spec = MODELS[role]
        self.model, self.device, self.task_id = spec["id"], f"cuda:{spec['device']}", None
        self.tok = AutoTokenizer.from_pretrained(self.model, revision=spec["revision"])
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        options = dict(revision=spec["revision"], dtype=torch.float16,
                       device_map={"": spec["device"]}, attn_implementation="sdpa")
        if spec["four_bit"]:
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.lm = AutoModelForCausalLM.from_pretrained(self.model, **options).eval()
        actual = {str(p.device) for p in self.lm.parameters()}
        if actual != {self.device}:
            raise RuntimeError(f"Wrong {role} placement: {actual}")
        quantized = sum(type(p).__name__ == "Params4bit" for p in self.lm.parameters())
        if spec["four_bit"] and not quantized:
            raise RuntimeError("Repair model did not load four-bit parameter tensors")
        record(dict(event="model_loaded", role=role, model=self.model, revision=spec["revision"],
                    device=self.device, four_bit=spec["four_bit"], quantized_parameters=quantized,
                    allocated_bytes=torch.cuda.memory_allocated(self.device)))

    def complete(self, messages, max_new_tokens, temperature, top_p, seed):
        torch = self.torch
        with torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tok(prompt, return_tensors="pt").to(self.device)
            count = int(inputs["input_ids"].shape[1])
            options = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature:
                options.update(do_sample=True, temperature=temperature, top_p=top_p or 1.0)
            else:
                options.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            torch.cuda.reset_peak_memory_stats(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            self.record(dict(event="generation_start", task_id=self.task_id, role=self.role,
                             prompt_tokens=count, max_new_tokens=max_new_tokens, free_bytes=free, total_bytes=total))
            started = time.monotonic()
            try:
                out = self.lm.generate(**inputs, **options)[0, count:]
                text = self.tok.decode(out, skip_special_tokens=True)
                tokens = int((out != self.tok.pad_token_id).sum())
                self.record(dict(event="generation_complete", task_id=self.task_id, role=self.role,
                                 prompt_tokens=count, output_tokens=tokens,
                                 peak_bytes=torch.cuda.max_memory_allocated(self.device),
                                 seconds=round(time.monotonic() - started, 3)))
                return text, count, tokens
            except Exception as error:
                self.record(dict(event="generation_failed", task_id=self.task_id, role=self.role,
                                 prompt_tokens=count, error_type=type(error).__name__, error=str(error),
                                 peak_bytes=torch.cuda.max_memory_allocated(self.device)))
                raise
