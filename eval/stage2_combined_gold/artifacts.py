from pathlib import Path
import hashlib,json
from eval.stage2_dual_3b.artifacts import ROOT,digest,write,export
from eval.stage2_combined_3b.artifacts import source_files as original_source_files

def source_files():
    return sorted(set(original_source_files()+list((ROOT/'eval/stage2_combined_gold').glob('*.py'))))
def runtime_hashes():return {str(p.relative_to(ROOT)):digest(p) for p in source_files()}
def verify_pack():
    path=ROOT/'_stage2_combined_gold_manifest.json'
    if not path.exists():return None
    raw=path.read_bytes();manifest=json.loads(raw)
    if manifest['experiment']!='stage2-combined-gold-3b-fp16-v1':raise ValueError('Wrong source pack')
    for name,sha in manifest['sha256'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or digest(ROOT/rel)!=sha:raise ValueError('Changed packaged input/source: '+name)
    return hashlib.sha256(raw).hexdigest()
