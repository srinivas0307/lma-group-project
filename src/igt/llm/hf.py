"""transformers backend for LLMClient (GPU; Kaggle). Single-conversation calls for the agent loop.
Batch generation for pools and parsers stays in eval/generate.py (HFBackend there)."""


class HFChatBackend:
    def __init__(self, model_id, dtype="float16", load_in_4bit=False, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model = model_id + (f"+adapter:{adapter}" if adapter else "")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(dtype=getattr(torch, dtype), device_map="auto" if torch.cuda.is_available() else None)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=getattr(torch, dtype))
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        if adapter:
            from peft import PeftModel
            self.lm = PeftModel.from_pretrained(self.lm, str(adapter)).eval()

    def complete(self, messages, max_new_tokens, temperature, top_p, seed):
        t = self.torch
        with t.inference_mode():
            if seed is not None:
                t.manual_seed(seed)
                t.cuda.manual_seed_all(seed)
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = self.tok(prompt, return_tensors="pt").to(self.lm.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature:
                kw.update(do_sample=True, temperature=temperature, top_p=top_p or 1.0)
            else:
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            out = self.lm.generate(**enc, **kw)[0, enc["input_ids"].shape[1]:]
            text = self.tok.decode(out, skip_special_tokens=True)
        return text, int(enc["input_ids"].shape[1]), int((out != self.tok.pad_token_id).sum())
