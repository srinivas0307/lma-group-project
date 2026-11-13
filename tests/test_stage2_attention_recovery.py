"""Test the GQA adapter's head expansion and preservation of attention semantics on CPU."""
import runpy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, data, device="cuda:0"):
        self.data = np.asarray(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.device = device

    def repeat_interleave(self, count, dim):
        return Tensor(np.repeat(self.data, count, axis=dim), self.device)


@pytest.fixture(params=["resume", "clean"])
def adapter_scope(request):
    if request.param == "clean":
        from eval.stage2_clean.attention import install_efficient_attention
        return {"install_efficient_attention": install_efficient_attention}
    builder = runpy.run_path(str(ROOT / "archive/notebooks/core/make_stage2_resume_notebook.py"))
    scope = {"__name__": "attention_adapter_test"}
    exec(builder["BOOTSTRAP"], scope)
    return scope


def numpy_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
    query, key, value = q.data, k.data, v.data
    if enable_gqa:
        count = query.shape[-3] // key.shape[-3]
        key, value = np.repeat(key, count, axis=-3), np.repeat(value, count, axis=-3)
    scores = query @ np.swapaxes(key, -1, -2) * (scale if scale is not None else query.shape[-1] ** -0.5)
    if is_causal:
        allowed = np.tril(np.ones(scores.shape[-2:], dtype=bool))
        scores = np.where(allowed, scores, -np.inf)
    if attn_mask is not None:
        scores = scores + attn_mask
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ value


@pytest.mark.parametrize("case", ["causal", "decode", "masked", "scaled"])
def test_grouped_heads_match_reference_and_preserve_mask_scale_and_causality(case, adapter_scope):
    rng = np.random.default_rng(7)
    query = Tensor(rng.normal(size=(1, 4, 1 if case == "decode" else 6, 8)))
    key, value = [Tensor(rng.normal(size=(1, 2, 6, 8))) for _ in range(2)]
    mask = np.array([[0, 0, 0, 0, 0, -np.inf]]) if case == "masked" else None
    options = dict(attn_mask=mask, is_causal=case == "causal", scale=0.25 if case == "scaled" else None,
                   dropout_p=0.0, enable_gqa=True)
    calls, contexts, events = [], [], []

    @contextmanager
    def context(backend):
        contexts.append(backend)
        yield

    def original(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        assert contexts[-1] == "efficient"
        return numpy_attention(q, k, v, **kwargs)

    functional = SimpleNamespace(scaled_dot_product_attention=original)
    scope = adapter_scope
    scope["install_efficient_attention"](functional, context, "efficient", events.append)
    actual = functional.scaled_dot_product_attention(query, key, value, **options)
    expected = numpy_attention(query, key, value, **options)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert calls[0][1].shape[-3] == calls[0][2].shape[-3] == query.shape[-3]
    assert calls[0][3] == dict(options, enable_gqa=False)
    assert calls[0][3]["attn_mask"] is mask
    assert events[0]["math_fallback_allowed"] is False


def test_adapter_does_not_change_cpu_attention(adapter_scope):
    scope = adapter_scope
    query = Tensor(np.ones((1, 4, 2, 8)), "cpu")
    key = value = Tensor(np.ones((1, 2, 2, 8)), "cpu")
    calls, events = [], []

    def original(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return "cpu-result"

    def no_cuda_context(backend):
        raise AssertionError("CPU call must not select a CUDA backend")

    functional = SimpleNamespace(scaled_dot_product_attention=original)
    scope["install_efficient_attention"](functional, no_cuda_context, "efficient", events.append)
    assert functional.scaled_dot_product_attention(query, key, value, enable_gqa=True) == "cpu-result"
    assert calls[0][1] is key and calls[0][2] is value
    assert calls[0][3]["enable_gqa"] is True
    assert events == []


def test_unsupported_kernel_is_reported_without_math_retry(adapter_scope):
    scope = adapter_scope
    tensor = Tensor(np.ones((1, 2, 2, 8)))
    calls, events = [], []

    @contextmanager
    def context(backend):
        assert backend == "efficient"
        yield

    def unsupported(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("efficient kernel unavailable")

    functional = SimpleNamespace(scaled_dot_product_attention=unsupported)
    scope["install_efficient_attention"](functional, context, "efficient", events.append)
    with pytest.raises(RuntimeError, match="efficient kernel unavailable"):
        functional.scaled_dot_product_attention(tensor, tensor, tensor)
    assert len(calls) == 1
    assert events[-1]["event"] == "attention_failure"
    assert events[-1]["math_fallback_allowed"] is False
