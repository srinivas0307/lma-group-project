"""Model loading/generation options are explicit; optional HF metadata is never required."""
from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from eval.stage2_clean.backend import Backend
from eval.stage2_clean.config import MODELS


def fake_inference(monkeypatch, quantized=True):
    calls = []
    torch = ModuleType("torch")
    torch.float16 = "float16"
    torch.inference_mode = nullcontext
    torch.manual_seed = lambda seed: None
    torch.cuda = SimpleNamespace(manual_seed_all=lambda seed: None, memory_allocated=lambda d: 10,
                                 reset_peak_memory_stats=lambda d: None, mem_get_info=lambda d: (100, 200),
                                 max_memory_allocated=lambda d: 20)

    class Batch(dict):
        def to(self, device):
            calls.append(("inputs", device))
            return self

    class Tokenizer:
        pad_token = None
        pad_token_id = 0
        eos_token = "eos"

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, text, **kwargs):
            return Batch(input_ids=np.array([[1, 2, 3]]))

        def decode(self, output, **kwargs):
            return "generated reply"

        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return Tokenizer()

    class Params4bit:
        pass

    class Model:
        def __init__(self, options):
            self.device = f"cuda:{options['device_map']['']}"
            self.quantized = "quantization_config" in options and quantized

        def parameters(self):
            parameter = Params4bit() if self.quantized else SimpleNamespace()
            parameter.device = self.device
            return iter([parameter])

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return np.array([[1, 2, 3, 4, 5]])

        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("model", name, kwargs))
            return Model(kwargs)

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForCausalLM = Model
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


@pytest.mark.parametrize("role", ["generator", "agent"])
def test_pinned_models_precision_device_generation_and_diagnostics(monkeypatch, role):
    calls = fake_inference(monkeypatch)
    events = []
    backend = Backend(role, events.append)
    backend.task_id = "example"
    assert backend.complete([{"role": "user", "content": "example"}], 768, 0.0, None, 0) == (
        "generated reply", 3, 2)
    spec = MODELS[role]
    tokenizer_call = next(c for c in calls if c[0] == "tokenizer")
    model_call = next(c for c in calls if c[0] == "model")
    assert tokenizer_call[2]["revision"] == model_call[2]["revision"] == spec["revision"]
    assert model_call[2]["dtype"] == "float16"
    assert model_call[2]["attn_implementation"] == "sdpa"
    assert model_call[2]["device_map"] == {"": spec["device"]}
    if role == "agent":
        assert model_call[2]["quantization_config"] == dict(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")
    else:
        assert "quantization_config" not in model_call[2]
    generation = next(c[1] for c in calls if c[0] == "generate")
    assert generation["max_new_tokens"] == 768 and generation["do_sample"] is False
    assert ("inputs", f"cuda:{spec['device']}") in calls
    assert events[-1]["event"] == "generation_complete" and events[-1]["task_id"] == "example"


def test_rejects_agent_that_was_not_quantized(monkeypatch):
    fake_inference(monkeypatch, quantized=False)
    with pytest.raises(RuntimeError, match="did not load four-bit"):
        Backend("agent", lambda event: None)
