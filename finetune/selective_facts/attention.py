"""The existing GQA/efficient SDPA policy, tested only on locally visible CUDA devices."""
import os
from eval.stage2_clean.attention import install_efficient_attention


def visible_devices(torch,devices=None):
    count=torch.cuda.device_count()
    selected=tuple(range(count)) if devices is None else tuple(devices)
    if not selected:raise RuntimeError('At least one visible CUDA GPU is required for sf1 attention')
    if len(set(selected))!=len(selected) or any(not isinstance(d,int) or d<0 or d>=count for d in selected):
        raise ValueError(f'Invalid local CUDA device indices {selected}; {count} device(s) visible')
    return selected


def configure_attention(torch,record,devices=None):
    from torch.nn.attention import SDPBackend,sdpa_kernel
    selected=visible_devices(torch,devices)
    record(dict(event='attention_visible_devices',devices=list(selected),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    original,attention=install_efficient_attention(torch.nn.functional,sdpa_kernel,SDPBackend.EFFICIENT_ATTENTION,record)
    try:
        # CUDA_VISIBLE_DEVICES renumbers exposed devices. A child seeing physical GPU 1
        # alone has local cuda:0; asking fork_rng for local index 1 crashes.
        with torch.inference_mode(),torch.random.fork_rng(devices=list(selected)):
            for index in selected:
                device=f'cuda:{index}'
                torch.manual_seed(0)
                query=torch.randn((1,4,32,128),device=device,dtype=torch.float16)
                key=torch.randn((1,2,32,128),device=device,dtype=torch.float16)
                value=torch.randn_like(key)
                mask=torch.zeros((32,32),device=device,dtype=torch.float16)
                mask[:,-1]=float('-inf')
                cases=[('causal_prefill',query,None,True),('decode',query[:,:,:1,:],None,False),('masked_prefill',query,mask,False)]
                for name,q,bias,causal in cases:
                    actual=attention(q,key,value,attn_mask=bias,is_causal=causal,enable_gqa=True)
                    with sdpa_kernel(SDPBackend.MATH):expected=original(q,key,value,attn_mask=bias,is_causal=causal,enable_gqa=True)
                    torch.testing.assert_close(actual,expected,atol=.005,rtol=.005)
                    torch.cuda.synchronize(device)
                    record(dict(event='kernel_self_test_passed',device=device,case=name,
                                max_absolute_error=float((actual-expected).abs().max().item()),rtol=.005,atol=.005))
                del query,key,value,mask,cases,q,bias,actual,expected
    except BaseException:
        torch.nn.functional.scaled_dot_product_attention=original
        raise
    torch.cuda.empty_cache()
    return selected


def probe_backward(torch,record):
    """Check the same installed attention wrapper supports the training backward pass."""
    selected=visible_devices(torch)
    with torch.random.fork_rng(devices=list(selected)),torch.enable_grad():
        for index in selected:
            device=f'cuda:{index}'
            q=torch.randn((1,4,8,128),device=device,dtype=torch.float16,requires_grad=True)
            k=torch.randn((1,2,8,128),device=device,dtype=torch.float16,requires_grad=True)
            v=torch.randn((1,2,8,128),device=device,dtype=torch.float16,requires_grad=True)
            output=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
            output.float().square().mean().backward()
            for tensor in [q,k,v]:
                if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
                    raise RuntimeError(f'Attention backward test failed on {device}')
            record(dict(event='kernel_backward_test_passed',device=device))
            del q,k,v,output
    torch.cuda.empty_cache()


def main():
    import json
    import torch
    events=[]
    def record(value):
        events.append(value);print(json.dumps(value),flush=True)
    selected=configure_attention(torch,record)
    probe_backward(torch,record)
    print('PASSED: visible-device attention forward/backward; local CUDA devices:',list(selected),flush=True)
if __name__=='__main__':main()
