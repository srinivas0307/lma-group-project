"""Kaggle inference setup and a tiny CPU LoRA probe before model downloads."""
import argparse
import importlib.metadata as metadata
import json
import subprocess
import sys
from packaging.version import Version


def prepare_optional_packages(version=metadata.version, run=subprocess.run, executable=sys.executable):
    """PEFT probes optional torchao even for ordinary FP16 LoRA modules.

    The notebook uses FP16 and bitsandbytes NF4, not torchao quantization. Remove
    only the known-incompatible optional installation in the disposable Kaggle
    session rather than installing a different quantization/CUDA stack.
    """
    try:
        installed=version('torchao')
    except metadata.PackageNotFoundError:
        return dict(torchao=None,action='already_absent')
    if Version(installed)<Version('0.16.0'):
        print(f'Removing optional torchao {installed}: incompatible with PEFT 0.21.2; this workflow does not use torchao.',flush=True)
        run([executable,'-m','pip','uninstall','-y','torchao'],check=True)
        try:remaining=version('torchao')
        except metadata.PackageNotFoundError:remaining=None
        if remaining is not None:raise RuntimeError(f'Incompatible torchao is still installed ({remaining}); restart in a fresh Kaggle session')
        return dict(torchao=None,action='removed_incompatible',previous_version=installed)
    return dict(torchao=installed,action='compatible_unchanged')


def lora_probe():
    import torch
    from peft import LoraConfig,get_peft_model
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.proj=torch.nn.Linear(4,4,bias=False)
        def forward(self,x):return self.proj(x)
    model=get_peft_model(Tiny(),LoraConfig(r=2,lora_alpha=2,target_modules=['proj']))
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable!=16 or tuple(model(torch.ones(1,4)).shape)!=(1,4):
        raise RuntimeError('Tiny LoRA injection/forward probe failed')
    return dict(probe='cpu_lora_injection_and_forward',trainable_parameters=trainable,status='passed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:print(json.dumps(prepare_optional_packages()),flush=True)
    print(json.dumps(lora_probe()),flush=True)

if __name__=='__main__':main()
