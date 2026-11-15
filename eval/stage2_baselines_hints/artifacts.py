from pathlib import Path
import hashlib,json
from eval.stage2_dual_3b.artifacts import ROOT,digest
from eval.stage2_repair_hints.artifacts import source_files as hints_source_files
EXPERIMENT='stage2-baselines-hints-3b-fp16-v1'
MANIFEST='_stage2_baselines_hints_manifest.json'

def source_files():
    return sorted(set(hints_source_files()+list((ROOT/'eval/stage2_baselines_hints').glob('*.py'))))
def runtime_hashes():return {str(p.relative_to(ROOT)):digest(p) for p in source_files()}
def verify_pack():
    path=ROOT/MANIFEST
    if not path.exists():return None
    raw=path.read_bytes();manifest=json.loads(raw)
    if manifest['experiment']!=EXPERIMENT:raise ValueError('Wrong source pack')
    for name,sha in manifest['sha256'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or digest(ROOT/rel)!=sha:raise ValueError('Changed packaged input/source: '+name)
    from eval.stage2_repair_hints.artifacts import verify_pack as hints_pack
    if not hints_pack():raise ValueError('The frozen repair-hints pack is missing from this source pack')
    return hashlib.sha256(raw).hexdigest()
