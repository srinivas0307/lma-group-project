"""Read-only compatibility check and discovery of the exact evaluated epoch-2 weights."""
import json
from pathlib import Path
import zipfile
from finetune.selective_facts.adapter import validate_adapter
from finetune.selective_facts.provenance import digest
SPEC=json.loads((Path(__file__).parent/'adapter_spec.json').read_text())

def validate_existing(folder):
    folder=Path(folder)
    checked=validate_adapter(folder)
    for name,expected in SPEC['sha256'].items():
        if digest(folder/name)!=expected:raise ValueError('Not the supplied full epoch-2 adapter: '+name)
    return checked

def discover(input_root,work):
    """Kaggle exposes datasets either extracted or as ZIP files. Never edit inputs."""
    input_root=Path(input_root);work=Path(work)
    for config in sorted(input_root.rglob('adapter_config.json')):
        try:validate_existing(config.parent)
        except (OSError,ValueError,KeyError):continue
        return config.parent
    for archive in sorted(input_root.rglob('selective_facts_1_5b.zip')):
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate result ZIP members')
            contents={}
            for name,expected in SPEC['sha256'].items():
                content=z.read('adapter/'+name)
                import hashlib
                if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('Wrong result ZIP adapter: '+name)
                contents[name]=content
            folder=work/'scopefix_existing_adapter';folder.mkdir(parents=True,exist_ok=True)
            for name,content in contents.items():
                path=folder/name
                if path.exists() and path.read_bytes()!=content:raise ValueError('Changed extracted adapter: '+name)
                path.write_bytes(content)
            validate_existing(folder);return folder
    raise ValueError('Attach the dataset made from selective_facts_1_5b.zip (the complete training output, not the smoke ZIP).')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('/kaggle/input'));p.add_argument('--work',type=Path,default=Path('/kaggle/working'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();folder=discover(a.input,a.work);a.out.write_text(str(folder));print('Verified unchanged epoch-2 adapter:',folder)
