import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hashes():
    paths=list((ROOT/'src/igt').rglob('*.py'))+list((ROOT/'finetune/selective_facts').glob('*.py'))
    paths += [ROOT/'eval/equivalence.py',ROOT/'eval/stage2_clean/attention.py',ROOT/'eval/contract_quality/harness.py',ROOT/'eval/contract_quality/build_suite.py',ROOT/'requirements.txt']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
def verify_data(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['context_sha256']!=digest(Path(__file__).parent/'context.py'):raise ValueError('Context implementation changed since data build')
    for name,expected in manifest['sha256'].items():
        if digest(folder/name)!=expected:raise ValueError(f'Training data changed: {name}')
    for name,expected in manifest['raw_data_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'Training CSV changed: {name}')
    return manifest
