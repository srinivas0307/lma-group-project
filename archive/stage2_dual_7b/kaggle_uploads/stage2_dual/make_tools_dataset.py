"""Create Kaggle-safe tool inputs without nested result archives; no runtime changes."""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def build():
    source=ROOT/'notebooks/kaggle/stage2_dual/stage2_dual_source.zip'
    training=ROOT/'kaggle_outputs/tool_use_v2/tool_use_train_1_5b.zip'
    with zipfile.ZipFile(source) as s,zipfile.ZipFile(training) as t:
        adapter={n:t.read(n) for n in t.namelist() if n.startswith('adapter/') and not n.endswith('/')}
        payload={n:s.read(n) for n in s.namelist() if not n.endswith('/')}
        if set(payload)&set(adapter):raise ValueError('Duplicate paths')
        payload.update(adapter)
        spec=json.loads(s.read('stage2_dual_source/eval/stage2_dual/adapter_specs.json'))['tools']['sha256']
        for name,sha in spec.items():
            if hashlib.sha256(adapter['adapter/'+name]).hexdigest()!=sha:raise ValueError('Different selected adapter')
        targets=[(HERE/'stage2_dual_tools_kaggle_dataset.zip',payload),(HERE/'tools_adapter/tool_use_train_1_5b.zip',adapter)]
        for target,files in targets:
            if any(n.endswith('.zip') for n in files):raise ValueError('Nested ZIP')
            target.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
                for n,b in sorted(files.items()):z.writestr(n,b)
            with zipfile.ZipFile(target) as z:
                if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Invalid output')
            print(target)
        info=dict(source_manifest_sha256=hashlib.sha256(s.read('stage2_dual_source/manifest.json')).hexdigest(),adapter_files={n:hashlib.sha256(b).hexdigest() for n,b in adapter.items()},note='Clean Kaggle inputs; no baseline/semantic result folders or nested ZIPs. Original runtime, weights and results unchanged.')
        (HERE/'tools_upload_info.json').write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':build()
