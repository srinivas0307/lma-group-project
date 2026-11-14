"""Prevent loading an old or unscreened adapter into the sf1 workflow."""
import json
from pathlib import Path
from .config import MODEL,REVISION
from .context import PROTOCOL
from .provenance import digest

def validate_adapter(folder):
    folder=Path(folder)
    metadata=json.loads((folder/'input_protocol.json').read_text())
    if metadata['protocol']!=PROTOCOL or metadata['base']!=MODEL or metadata['revision']!=REVISION:raise ValueError('Adapter protocol/base revision differs')
    if metadata['context_sha256']!=digest(Path(__file__).parent/'context.py'):raise ValueError('Adapter was trained with a different facts implementation')
    if (folder/'igt_prompt_version.txt').read_text().strip()!=PROTOCOL:raise ValueError('Wrong prompt marker')
    selection=json.loads((folder/'selection.json').read_text())
    if selection['smoke'] or not selection['eligible']:raise ValueError('Adapter did not pass full development screening; inspect selection.json')
    for name,expected in selection['selected']['adapter_sha256'].items():
        if Path(name).name!=name:raise ValueError('Invalid adapter provenance filename')
        if digest(folder/name)!=expected:raise ValueError(f'Adapter differs from evaluated checkpoint: {name}')
    config=json.loads((folder/'adapter_config.json').read_text())
    if config['base_model_name_or_path']!=MODEL:raise ValueError('Wrong base model')
    names=['input_protocol.json','selection.json','igt_prompt_version.txt','adapter_config.json','adapter_model.safetensors']
    return {name:digest(folder/name) for name in names}
