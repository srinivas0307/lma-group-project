"""Resolve the two exact selected adapters without relabeling or editing their metadata."""
import hashlib
import json
from pathlib import Path
import zipfile
from .artifacts import digest,write
SPECS=json.loads((Path(__file__).parent/'adapter_specs.json').read_text())

def validate(method,folder):
    folder=Path(folder);spec=SPECS[method]
    if method=='selective':
        from finetune.selective_facts_scopefix.adapter import validate_existing
        validate_existing(folder)
    else:
        from finetune.tool_use.adapter import validate as validate_tool
        validate_tool(folder,'1_5b')
    for n,sha in spec['sha256'].items():
        if digest(folder/n)!=sha:raise ValueError('Different selected adapter: '+n)
    selection=json.loads((folder/'selection.json').read_text())
    if selection['smoke'] or not selection['eligible'] or selection['selected']['name']!=spec['selected']:raise ValueError('Unscreened or smoke adapter')
    return dict(spec['sha256'])

def discover(method,input_root,work):
    input_root=Path(input_root);work=Path(work);spec=SPECS[method]
    for config in sorted(input_root.rglob('adapter_config.json')):
        try:validate(method,config.parent)
        except (OSError,ValueError,KeyError):continue
        return config.parent
    for archive in sorted(input_root.rglob(spec['archive_name'])):
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate ZIP members')
            data={n:z.read('adapter/'+n) for n in spec['sha256']}
            if any(hashlib.sha256(data[n]).hexdigest()!=sha for n,sha in spec['sha256'].items()):raise ValueError('Wrong adapter ZIP')
            folder=work/('stage2_dual_adapter_'+method);folder.mkdir(parents=True,exist_ok=True)
            for n,b in data.items():
                p=folder/n
                if p.exists() and p.read_bytes()!=b:raise ValueError('Changed extracted adapter')
                p.write_bytes(b)
            validate(method,folder);return folder
    raise ValueError('Attach/place '+spec['archive_name']+' or its complete extracted adapter folder')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--method',choices=SPECS,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();folder=discover(a.method,a.input,a.work);a.out.write_text(str(folder));print('Verified selected',a.method,'adapter:',folder)
