"""Independent source manifest for the repair-development package."""
import hashlib
import json
from pathlib import Path
from eval.stage2_dual_3b.artifacts import source_files as base_source_files
ROOT=Path(__file__).resolve().parents[2]


def source_files():
    return sorted(set(base_source_files()+list((ROOT/'eval/stage2_repair_dev').glob('*.py'))+
        [ROOT/'eval/stage2_independent_v2/environment.py',ROOT/'eval/stage2_independent_v2/__init__.py']))


def runtime_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}


def verify_pack():
    manifest=ROOT/'_repair_dev_manifest.json'
    if not manifest.exists():
        return None
    content=json.loads(manifest.read_text())
    if content['experiment']!='stage2-3b-repair-dev-v1':
        raise ValueError('Wrong repair development package')
    for name,sha in content['sha256'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid manifest path')
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:
            raise ValueError('Changed source: '+name)
    return hashlib.sha256(manifest.read_bytes()).hexdigest()
