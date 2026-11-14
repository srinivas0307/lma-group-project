from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[2]
DEPENDENCIES=['src/igt','eval/stage2_dual','eval/stage2_selective','finetune/selective_facts','finetune/selective_facts_scopefix','finetune/selective_facts_correction','finetune/tool_use','eval/stage2_tools']
EXTRA=['eval/equivalence.py','eval/stage2_clean/attention.py','eval/contract_quality/context.py','eval/contract_quality/harness.py','eval/contract_quality/build_suite.py','requirements.txt']

def source_files():
    paths=[]
    for name in DEPENDENCIES:
        paths += [p for p in (ROOT/name).rglob('*') if p.is_file() and (p.suffix=='.py' or (p.suffix=='.json' and p.name in ['adapter_specs.json','adapter_spec.json','adapter.json']))]
    paths += [ROOT/n for n in EXTRA]
    return sorted(set(paths))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def runtime_hashes():return {str(p.relative_to(ROOT)):digest(p) for p in source_files()}
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value,indent=2,default=str)+'\n');temp.replace(path)
def export(out,complete=False):
    out=Path(out);target=out.with_suffix('.zip');temp=Path(str(target)+'.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        seals={}
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix!='.tmp':
                name=str(p.relative_to(out));z.write(p,name);seals[name]=digest(p)
        z.writestr('bundle.json',json.dumps(dict(complete=complete,sha256=seals)))
    temp.replace(target)
