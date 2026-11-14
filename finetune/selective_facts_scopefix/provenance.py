from pathlib import Path
from finetune.selective_facts.provenance import ROOT,digest,hashes as original_hashes

def hashes():
    result=original_hashes()
    result.update({str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file() and p.suffix in ['.py','.json']})
    return result
