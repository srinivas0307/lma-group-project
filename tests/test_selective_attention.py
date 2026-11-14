"""Reproduce one-visible-GPU isolation without requiring local Torch or CUDA."""
import contextlib
import importlib
import sys
import types
from pathlib import Path
import numpy as np
import pytest

class Tensor:
    def __init__(self,data,device='cuda:0'):
        self.data=np.asarray(data);self.device=device;self.dtype='float16'
    @property
    def shape(self):return self.data.shape
    def repeat_interleave(self,n,dim):return Tensor(np.repeat(self.data,n,axis=dim),self.device)
    def __getitem__(self,key):return Tensor(self.data[key],self.device)
    def __setitem__(self,key,value):self.data[key]=value
    def __sub__(self,other):return Tensor(self.data-other.data,self.device)
    def abs(self):return Tensor(np.abs(self.data),self.device)
    def max(self):return Tensor(np.max(self.data),self.device)
    def item(self):return self.data.item()

class IsolatedTorch(types.ModuleType):
    def __init__(self,count,fail=False):
        super().__init__('torch');self.count=count;self.forks=[];self.allocations=[];self.restored=False;self.float16='float16'
        self.nn=types.ModuleType('torch.nn');self.nn.functional=types.SimpleNamespace()
        def original(q,k,v,**kwargs):
            if fail:raise RuntimeError('simulated GPU kernel failure')
            return Tensor(np.zeros(q.shape),q.device)
        self.original=original;self.nn.functional.scaled_dot_product_attention=original
        self.cuda=types.SimpleNamespace(device_count=lambda:self.count,synchronize=lambda device:self.check(device),empty_cache=lambda:None)
        @contextlib.contextmanager
        def fork_rng(devices):
            if any(i<0 or i>=count for i in devices):raise IndexError('tuple index out of range')
            self.forks.append(list(devices))
            try:yield
            finally:self.restored=True
        self.random=types.SimpleNamespace(fork_rng=fork_rng)
        self.testing=types.SimpleNamespace(assert_close=lambda a,b,atol,rtol:np.testing.assert_allclose(a.data,b.data,atol=atol,rtol=rtol))
    def check(self,device):
        if int(str(device).split(':')[1])>=self.count:raise IndexError('Invalid CUDA device')
    def randn(self,shape,device,dtype):
        self.check(device);self.allocations.append(device)
        return Tensor(np.zeros(shape),device)
    def randn_like(self,tensor):return self.randn(tensor.shape,tensor.device,self.float16)
    def zeros(self,shape,device,dtype):return self.randn(shape,device,dtype)
    def manual_seed(self,seed):pass
    def inference_mode(self):return contextlib.nullcontext()


def install_modules(monkeypatch,torch):
    attention=types.ModuleType('torch.nn.attention')
    attention.SDPBackend=types.SimpleNamespace(EFFICIENT_ATTENTION='efficient',MATH='math')
    attention.sdpa_kernel=lambda backend:contextlib.nullcontext()
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'torch.nn',torch.nn)
    monkeypatch.setitem(sys.modules,'torch.nn.attention',attention)


@pytest.mark.parametrize('count,mask',[(1,'0'),(1,'1'),(2,'0,1')])
def test_attention_self_test_uses_local_visible_device_indices(monkeypatch,count,mask):
    from finetune.selective_facts.attention import configure_attention
    torch=IsolatedTorch(count);install_modules(monkeypatch,torch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',mask)
    events=[];selected=configure_attention(torch,events.append)
    assert selected==tuple(range(count))
    assert torch.forks==[list(range(count))] and torch.restored
    assert set(torch.allocations)=={f'cuda:{i}' for i in range(count)}
    assert len([e for e in events if e['event']=='kernel_self_test_passed'])==3*count
    assert events[0]['cuda_visible_devices']==mask


def test_attention_failure_restores_original_function_and_rng(monkeypatch):
    from finetune.selective_facts.attention import configure_attention
    torch=IsolatedTorch(1,fail=True);install_modules(monkeypatch,torch)
    with pytest.raises(RuntimeError,match='simulated GPU'):configure_attention(torch,lambda e:None)
    assert torch.nn.functional.scaled_dot_product_attention is torch.original
    assert torch.restored


def test_invalid_or_absent_devices_fail_before_installing_attention(monkeypatch):
    from finetune.selective_facts.attention import visible_devices,configure_attention
    with pytest.raises(RuntimeError,match='At least one'):visible_devices(IsolatedTorch(0))
    for indices in [(1,),(-1,),(0,0)]:
        with pytest.raises(ValueError,match='Invalid local'):visible_devices(IsolatedTorch(1),indices)


def test_training_and_evaluation_use_isolation_safe_attention():
    import ast
    root=Path(__file__).resolve().parents[1]
    for name in ['train.py','evaluate.py']:
        tree=ast.parse((root/'finetune/selective_facts'/name).read_text())
        imports=[n for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and any(a.name=='configure_attention' for a in n.names)]
        assert len(imports)==1 and imports[0].module=='attention' and imports[0].level==1
    book=__import__('nbformat').read(root/'notebooks/kaggle/selective_facts/selective_facts_train.ipynb',as_version=4)
    setup=book.cells[2].source
    assert "for physical_gpu in ['0','1']" in setup
    assert "'finetune.selective_facts.attention'" in setup
    assert "CUDA_VISIBLE_DEVICES=physical_gpu" in setup
