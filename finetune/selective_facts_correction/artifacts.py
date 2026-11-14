import json
from pathlib import Path
import zipfile
from finetune.selective_facts.provenance import ROOT,digest,write
from finetune.selective_facts_scopefix.provenance import hashes as inherited_hashes

def hashes():
    result=inherited_hashes()
    result.update({str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('*') if p.is_file() and p.suffix in ['.py','.json']})
    return result

def export(out,complete=False):
    out=Path(out);target=out.with_suffix('.zip');temp=Path(str(target)+'.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        seals={}
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix!='.tmp':
                name=str(p.relative_to(out));z.write(p,name);seals[name]=digest(p)
        z.writestr('bundle.json',json.dumps(dict(complete=complete,sha256=seals)))
    temp.replace(target)
