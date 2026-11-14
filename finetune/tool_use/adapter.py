import json
from pathlib import Path
from .config import MODELS,PROTOCOL
from finetune.selective_facts.provenance import digest

def validate(path,size):
    path=Path(path);meta=json.loads((path/'tool_protocol.json').read_text())
    if meta['protocol']!=PROTOCOL or meta['size']!=size or meta['base']!=MODELS[size]:raise ValueError('Wrong tool adapter protocol or model size')
    if meta.get('smoke'):raise ValueError('Smoke adapter cannot be used for evaluation')
    for name,expected in meta['sha256'].items():
        if digest(path/name)!=expected:raise ValueError('Adapter hash mismatch: '+name)
    if not {'adapter_config.json','adapter_model.safetensors'}<=meta['sha256'].keys():raise ValueError('Missing adapter seals')
    return meta
