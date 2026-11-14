"""vLLM backend for LLMClient (extension #7, id `vllm`; GPU only, optional dependency: requirements-vllm.txt).

Same interface as igt.llm.hf.HFChatBackend:
    b = VLLMChatBackend("Qwen/Qwen2.5-7B-Instruct", dtype="float16", load_in_4bit=True, adapter=None)
    text, prompt_tokens, output_tokens = b.complete(messages, max_new_tokens, temperature, top_p, seed)

Greedy outputs can differ from transformers (other kernels), so a vLLM run is its own version: the model name is
prefixed "vllm:" (it enters every run record and the LLM cache key, so cached transformers replies are never
reused), and the extension runners add "+vllm" to the recorded agent version and "_vllm" to the system name.
Existing results stay on transformers.

Settings (environment): IGT_VLLM_GPU_UTIL (default 0.85; two models in one process, e.g. Stage 2 generator + agent,
need about 0.4 each), IGT_VLLM_MAX_LEN (default 16384), IGT_VLLM_MAX_LORA_RANK (default 64), IGT_VLLM_TP
(tensor-parallel size, default 1).
"""
import os


def _default_llm_factory(**kw):
    from vllm import LLM
    return LLM(**kw)


class VLLMChatBackend:
    _llm_factory = staticmethod(_default_llm_factory)     # tests replace this with a stub

    def __init__(self, model_id, dtype="float16", load_in_4bit=False, adapter=None):
        self.model_id, self.adapter = model_id, adapter
        self.model = "vllm:" + model_id + (f"+adapter:{adapter}" if adapter else "")
        kw = dict(model=model_id, dtype=dtype, seed=0, enforce_eager=False,
                  gpu_memory_utilization=float(os.environ.get("IGT_VLLM_GPU_UTIL", "0.85")),
                  max_model_len=int(os.environ.get("IGT_VLLM_MAX_LEN", "16384")),
                  tensor_parallel_size=int(os.environ.get("IGT_VLLM_TP", "1")))
        if load_in_4bit:
            kw.update(quantization="bitsandbytes")
        if adapter:
            kw.update(enable_lora=True, max_lora_rank=int(os.environ.get("IGT_VLLM_MAX_LORA_RANK", "64")))
        self.llm = self._llm_factory(**kw)
        self.tok = self.llm.get_tokenizer()
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("adapter", 1, str(adapter))

    def _sampling(self, max_new_tokens, temperature, top_p, seed):
        from vllm import SamplingParams
        if temperature:
            return SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p or 1.0, seed=seed)
        return SamplingParams(max_tokens=max_new_tokens, temperature=0.0, seed=seed)

    def complete(self, messages, max_new_tokens, temperature, top_p, seed):
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kw = dict(sampling_params=self._sampling(max_new_tokens, temperature, top_p, seed), use_tqdm=False)
        if self.lora is not None:
            kw["lora_request"] = self.lora
        out = self.llm.generate([prompt], **kw)[0]
        c = out.outputs[0]
        return c.text, len(out.prompt_token_ids), len(c.token_ids)


def backend_class(name):
    """'hf' -> HFChatBackend, 'vllm' -> VLLMChatBackend (imported lazily; neither is needed on CPU)."""
    if name == "vllm":
        return VLLMChatBackend
    if name == "hf":
        from igt.llm.hf import HFChatBackend
        return HFChatBackend
    raise ValueError(f"unknown backend {name}")
