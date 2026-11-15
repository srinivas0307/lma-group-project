"""Create clean single-upload Kaggle datasets for each 3B FP16 repair method."""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def build():
    source=ROOT/'notebooks/kaggle/stage2_combined_3b/stage2_combined_3b_source.zip'
    specs=json.loads((ROOT/'eval/stage2_combined_3b/adapter_specs.json').read_text())
    outputs={}
    for method,relative in [('selective','kaggle_outputs/training/selective_facts_v1/selective_facts_1_5b.zip'),('tools','kaggle_outputs/tool_use_v2/tool_use_train_1_5b.zip')]:
        with zipfile.ZipFile(source) as s,zipfile.ZipFile(ROOT/relative) as t:
            data={n:s.read(n) for n in s.namelist() if not n.endswith('/')}
            for name,sha in specs[method]['sha256'].items():
                value=t.read('adapter/'+name)
                if hashlib.sha256(value).hexdigest()!=sha:raise ValueError('Selected adapter mismatch')
                data['adapter/'+name]=value
            if any(n.endswith('.zip') or n.startswith(('baseline/','semantic/')) for n in data):raise ValueError('Nested archive or unnecessary results')
            target=HERE/f'stage2_combined_3b_{method}_kaggle_dataset.zip'
            with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
                for n,b in sorted(data.items()):z.writestr(n,b)
            with zipfile.ZipFile(target) as z:
                if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Invalid output')
            outputs[method]=dict(path=str(target.relative_to(ROOT)),source_manifest_sha256=hashlib.sha256(s.read('stage2_combined_3b_source/manifest.json')).hexdigest(),adapter_sha256=specs[method]['sha256'])
            print(target,flush=True)
    (HERE/'upload_info.json').write_text(json.dumps(outputs,indent=2)+'\n')
if __name__=='__main__':build()
