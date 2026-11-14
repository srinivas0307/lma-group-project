"""One interface for every LLM call (README Phase 4): LLMClient + backends + cache.

    client = LLMClient(backend, cache_dir="/tmp/llm_cache", token_budget=20_000)
    r = client.generate(messages, max_new_tokens=512, seed=1)      # -> LLMResponse
    r.text, r.prompt_tokens, r.output_tokens, r.cached

Backends implement `complete(messages, max_new_tokens, temperature, top_p, seed) -> (text, prompt_tokens, output_tokens)`
and have a `model` name:
- ScriptedBackend: replays pre-written responses (agent tests on the laptop, no GPU). A response can be a string or
  a function(messages) -> string. It records every call and fails loudly if the script runs out.
- HFBackend (igt.llm.hf): transformers on a GPU (Kaggle); imported lazily so the laptop needs no torch.

The cache key is (model, messages, max_new_tokens, temperature, top_p, seed); a hit returns the stored text, so
reruns are free and identical. Greedy calls (temperature 0) ignore the seed in the key.
"""
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    output_tokens: int
    cached: bool = False
    latency_s: float = 0.0


class ScriptExhausted(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class ScriptedBackend:
    """Replays responses in order. Each item: a string, or callable(messages) -> string."""

    def __init__(self, responses, model="scripted"):
        self.responses = list(responses)
        self.model = model
        self.calls = []

    def complete(self, messages, max_new_tokens, temperature, top_p, seed):
        if len(self.calls) >= len(self.responses):
            raise ScriptExhausted(f"script has {len(self.responses)} responses; call {len(self.calls) + 1} has none")
        r = self.responses[len(self.calls)]
        self.calls.append([dict(m) for m in messages])
        text = r(messages) if callable(r) else r
        n_in = sum(len(m["content"]) for m in messages) // 4
        return text, n_in, max(1, len(text) // 4)


def _key(model, messages, max_new_tokens, temperature, top_p, seed):
    payload = dict(model=model, messages=messages, max_new_tokens=max_new_tokens, temperature=temperature,
                   top_p=top_p, seed=None if not temperature else seed)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class LLMClient:
    def __init__(self, backend, cache_dir=None, token_budget=None):
        self.backend = backend
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token_budget = token_budget
        self.tokens_used = 0
        self.n_calls = 0
        self.n_cache_hits = 0

    @property
    def model(self):
        return self.backend.model

    def generate(self, messages, max_new_tokens=512, temperature=0.0, top_p=None, seed=None) -> LLMResponse:
        if self.token_budget is not None and self.tokens_used >= self.token_budget:
            raise BudgetExceeded(f"token budget {self.token_budget} used up")
        key = _key(self.model, messages, max_new_tokens, temperature, top_p, seed)
        path = self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None
        self.n_calls += 1
        if path and path.exists():
            d = json.loads(path.read_text())
            self.n_cache_hits += 1
            self.tokens_used += d["prompt_tokens"] + d["output_tokens"]
            return LLMResponse(d["text"], self.model, d["prompt_tokens"], d["output_tokens"], cached=True)
        t0 = time.time()
        text, n_in, n_out = self.backend.complete(messages, max_new_tokens, temperature, top_p, seed)
        resp = LLMResponse(text, self.model, int(n_in), int(n_out), cached=False, latency_s=round(time.time() - t0, 3))
        self.tokens_used += resp.prompt_tokens + resp.output_tokens
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(text=text, prompt_tokens=resp.prompt_tokens, output_tokens=resp.output_tokens,
                                           model=self.model), ensure_ascii=False))
            tmp.replace(path)
        return resp

    def stats(self):
        return dict(model=self.model, calls=self.n_calls, cache_hits=self.n_cache_hits, tokens_used=self.tokens_used,
                    token_budget=self.token_budget)
