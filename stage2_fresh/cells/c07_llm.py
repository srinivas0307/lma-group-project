# ===== Model loading + chat generation (greedy, token-accounted) =====
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_TF_NEW = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)
print("torch", torch.__version__, "| transformers", transformers.__version__, "| GPUs:", torch.cuda.device_count())


def _dtype_kw(dt):
    return {"dtype": dt} if _TF_NEW else {"torch_dtype": dt}


class ChatModel:
    def __init__(self, base_id, device, revision=None, adapter=None, load_4bit=False, name=None):
        self.name = name or base_id
        tok_src = str(adapter) if adapter else base_id
        self.tok = AutoTokenizer.from_pretrained(tok_src, **({} if adapter else {"revision": revision}))
        kw = dict(revision=revision, device_map={"": device}, attn_implementation=ATTN_IMPL, **_dtype_kw(torch.float16))
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_use_double_quant=False)
        model = AutoModelForCausalLM.from_pretrained(base_id, **kw)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
        self.model = model
        self.device = device
        self.calls = 0

    @torch.inference_mode()
    def chat(self, messages, max_new_tokens=1024, tools=None):
        kw = {"tools": tools} if tools else {}
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                  top_p=None, top_k=None, repetition_penalty=REPETITION_PENALTY,
                                  pad_token_id=self.tok.eos_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True)
        self.calls += 1
        return text, {"prompt_tokens": int(enc["input_ids"].shape[1]), "completion_tokens": int(new.shape[0]),
                      "truncated": bool(new.shape[0] >= max_new_tokens)}


N_GPU = torch.cuda.device_count()
PARSER_DEVICE = "cuda:0"
CODER_DEVICE = "cuda:1" if N_GPU > 1 else "cuda:0"

t0 = time.time()
PARSER_LM = ChatModel(PARSER_BASE, PARSER_DEVICE, revision=PARSER_REVISION, adapter=PARSER_ADAPTER, load_4bit=True,
                      name=f"{PARSER_BASE}+{PARSER_PROTOCOL}")
CODER_LM = ChatModel(CODER_BASE, CODER_DEVICE, load_4bit=False, name=f"{CODER_BASE} fp16")
print(f"models loaded in {time.time() - t0:.0f}s | parser on {PARSER_DEVICE} (NF4 + adapter) | generator/repair on {CODER_DEVICE} (FP16)")
