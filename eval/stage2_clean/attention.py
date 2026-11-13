"""Fused CUDA attention with explicit GQA expansion and an actual GPU self-test."""

def install_efficient_attention(functional, kernel_context, efficient_backend, record):
    """Expand grouped K/V heads explicitly, then require the fused CUDA kernel."""
    original = functional.scaled_dot_product_attention
    reported = set()

    def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                  *, scale=None, enable_gqa=False):
        if not str(query.device).startswith("cuda:"):
            return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        original_key_shape = list(key.shape)
        if enable_gqa:
            heads, kv_heads = query.shape[-3], key.shape[-3]
            if kv_heads != value.shape[-3] or heads % kv_heads:
                raise ValueError("Invalid grouped-query head counts")
            # This is the expansion in PyTorch's documented GQA definition.
            # Native enable_gqa=True cannot use the efficient CUDA kernel in torch 2.11.
            key = key.repeat_interleave(heads // kv_heads, dim=-3)
            value = value.repeat_interleave(heads // kv_heads, dim=-3)
        stage = "prefill" if query.shape[-2] > 1 else "decode"
        signature = (str(query.device), query.shape[-3], stage, attn_mask is not None)
        details = dict(device=str(query.device), query_shape=list(query.shape),
                       original_key_shape=original_key_shape, expanded_key_shape=list(key.shape),
                       dtype=str(query.dtype), native_gqa_expanded=bool(enable_gqa),
                       stage=stage, has_mask=attn_mask is not None, is_causal=bool(is_causal),
                       backend="EFFICIENT_ATTENTION", math_fallback_allowed=False)
        if signature not in reported:
            record(dict(event="attention_shape", **details))
            reported.add(signature)
        try:
            with kernel_context(efficient_backend):
                return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                is_causal=is_causal, scale=scale, enable_gqa=False)
        except Exception as error:
            record(dict(event="attention_failure", error=str(error), **details))
            raise

    functional.scaled_dot_product_attention = attention
    return original, attention


def configure_attention(torch, record):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original, attention = install_efficient_attention(
        torch.nn.functional, sdpa_kernel, SDPBackend.EFFICIENT_ATTENTION, record)
    # Test the actual Kaggle kernels before downloading/loading model weights.
    # Tiny math-reference calls are confined to this self-test, not model inference.
    with torch.inference_mode(), torch.random.fork_rng(devices=[0, 1]):
        for device in ("cuda:0", "cuda:1"):
            torch.manual_seed(0)
            query = torch.randn((1, 4, 32, 128), device=device, dtype=torch.float16)
            key = torch.randn((1, 2, 32, 128), device=device, dtype=torch.float16)
            value = torch.randn_like(key)
            mask = torch.zeros((32, 32), device=device, dtype=torch.float16)
            mask[:, -1] = float("-inf")
            cases = [("causal_prefill", query, None, True),
                     ("decode", query[:, :, :1, :], None, False),
                     ("masked_prefill", query, mask, False)]
            for name, q, bias, causal in cases:
                actual = attention(q, key, value, attn_mask=bias, is_causal=causal, enable_gqa=True)
                with sdpa_kernel(SDPBackend.MATH):
                    expected = original(q, key, value, attn_mask=bias, is_causal=causal, enable_gqa=True)
                torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
                torch.cuda.synchronize(device)
                record(dict(event="kernel_self_test_passed", device=device, case=name,
                            max_absolute_error=float((actual - expected).abs().max().item()),
                            rtol=0.005, atol=0.005))
            del query, key, value, mask, cases, q, bias, actual, expected
    torch.cuda.empty_cache()
