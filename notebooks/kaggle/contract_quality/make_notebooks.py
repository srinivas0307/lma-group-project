"""Build portable pilot notebooks. Payload includes source and draft suite; excludes model weights."""
import base64
import hashlib
import io
import json
import sys
import textwrap
import zipfile
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def payload():
    paths=list((ROOT/'src/igt').rglob('*.py'))+list((ROOT/'eval/contract_quality').glob('*.py'))
    paths += [ROOT/'eval/equivalence.py',ROOT/'eval/stage2_clean/attention.py',ROOT/'eval/stage2_clean/__init__.py',ROOT/'requirements.txt']
    paths += [p for p in (ROOT/'benchmark/contract_quality_v1').rglob('*') if p.is_file()]
    raw=json.dumps({str(p.relative_to(ROOT)):p.read_text() for p in sorted(paths)},sort_keys=True).encode()
    import zlib
    return base64.b64encode(zlib.compress(raw,9)).decode(),hashlib.sha256(raw).hexdigest()

def build():
    encoded,sha=payload()
    for size in ['1_5b','3b']:
        cells=[nb.v4.new_markdown_cell(f'''# Contract-quality pilot: saved {size} adapter
Exploratory synthetic **draft**, open development only. No training and no Stage 2 generation/repair.
Select **GPU T4 x2**, Internet on. Attach a private Kaggle dataset containing the original adapter folders
`qwen25_1_5b/adapter/` and `qwen25_3b/adapter/` (all files). Automatic discovery requires exactly one matching adapter.
The notebook embeds its code and suite: **no GitHub push is needed**. A CPU-validated notebook is not a GPU execution guarantee.
A four-call smoke run precedes the 160-call paired experiment. Do not change the method after viewing selection labels.''')]
        bootstrap='''import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path
WORK=Path('/kaggle/working')
WORK.mkdir(parents=True,exist_ok=True)
'''
        bootstrap+=f"SIZE={size!r}\nPAYLOAD_SHA={sha!r}\nPAYLOAD=(\n"+'\n'.join(repr(s) for s in textwrap.wrap(encoded,100))+')\n'
        bootstrap+='''raw=zlib.decompress(base64.b64decode(PAYLOAD))
assert hashlib.sha256(raw).hexdigest()==PAYLOAD_SHA
ROOT=WORK/('cq_source_'+PAYLOAD_SHA[:12])
for name,source in json.loads(raw).items():
    relative=Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    path=ROOT/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==source, f'Changed embedded file: {path}'
    else:path.write_text(source)
spec=json.loads((ROOT/'benchmark/contract_quality_v1/adapters.json').read_text())[SIZE]
matches=[]
for path in Path('/kaggle/input').rglob('adapter_config.json'):
    try:
        if json.loads(path.read_text())['base_model_name_or_path']==spec['model']:
            matches.append(path.parent)
    except (KeyError,ValueError):pass
assert len(matches)==1, f'Expected one adapter for {SIZE}; found {matches}. Attach the original adapter dataset.'
ADAPTER=matches[0]
for name,expected in spec['sha256'].items():
    assert hashlib.sha256((ADAPTER/name).read_bytes()).hexdigest()==expected, f'Wrong adapter file: {name}'
print('Payload:', PAYLOAD_SHA, 'Adapter:', ADAPTER)
'''
        cells.append(nb.v4.new_code_cell(bootstrap))
        cells.append(nb.v4.new_code_cell('''# Install CPU/verifier packages separately; use only child processes for heavy imports.
PACKAGES=WORK/'cq_packages'
subprocess.run([sys.executable,'-m','pip','install','-q','--upgrade','--target',str(PACKAGES),'-r',str(ROOT/'requirements.txt')],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.20.2','peft==0.21.2','accelerate==1.15.0','tokenizers==0.23.3','safetensors==0.8.0','huggingface-hub==1.33.0'],check=True)
ENV=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'src',PACKAGES])),TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
probe=subprocess.run([sys.executable,'-c',"import torch; assert torch.__version__ == '2.11.0+cu130', f'Expected torch 2.11.0+cu130, found {torch.__version__}'; print(torch.__version__); assert torch.cuda.device_count()==2"],cwd=ROOT,env=ENV,check=True)
print('Environment ready. If the pinned torch check fails, stop and share the setup log; do not silently change runtime.')
'''))
        cells.append(nb.v4.new_code_cell('''def run_pilot(smoke):
    name='cq_'+SIZE+('_smoke' if smoke else '_pilot')
    out=WORK/name
    command=[sys.executable,'-m','eval.contract_quality.pilot','--size',SIZE,'--adapter',str(ADAPTER),'--out',str(out)]
    if smoke:command.append('--smoke')
    log=WORK/(name+'.log')
    with log.open('w') as f:
        process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');f.write(line);f.flush()
        code=process.wait()
    assert code==0, f'Run failed: {log}. Download the log and any partial ZIP; do not score an incomplete run.'
    done=json.loads((out/'completion.json').read_text())
    assert done['complete'] and done['records']==(4 if smoke else 160)
    print('Bundle:',str(out)+'.zip')
run_pilot(smoke=True)
'''))
        cells.append(nb.v4.new_code_cell('''run_pilot(smoke=False)
from IPython.display import FileLink, display
display(FileLink(str(WORK/('cq_'+SIZE+'_pilot.zip'))))
'''))
        cells.append(nb.v4.new_markdown_cell('''Download `cq_*_pilot.zip` and the executed notebook. Put them in
`kaggle_outputs/contract_quality/v1/` and `notebooks/executed/contract_quality/v1/` respectively.
Run both adapter notebooks, then attach their full ZIP outputs to `cq_collect_results.ipynb` with accelerator None. The collector produces `cq_comparison_bundle.zip`; local scoring is also documented in `docs/contract_quality/RUNBOOK.md`.
The smoke ZIP is only an infrastructure check, not an experiment result.'''))
        notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        nb.validate(notebook)
        for cell in cells:
            if cell.cell_type=='code':compile(cell.source,'notebook','exec')
        nb.write(notebook,HERE/f'cq_{size}_pilot.ipynb')
    collector_bootstrap=bootstrap.split("spec=json.loads",1)[0]
    collector=nb.v4.new_notebook(cells=[
        nb.v4.new_markdown_cell("# Contract-quality collector (CPU)\nAttach the completed outputs of both GPU notebooks from this pack. Select **None** for accelerator and Internet on. This evaluates contracts against known correct and wrong transformations; it does not call a model."),
        nb.v4.new_code_cell(collector_bootstrap),
        nb.v4.new_code_cell('''# Use a venv so sandboxed python -I loads the same pinned pandas/numpy as the scorer.
VENV=WORK/'cq_cpu_env'
subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(VENV)],check=True)
PYTHON=VENV/'bin/python'
subprocess.run([str(PYTHON),'-m','pip','install','-q','-r',str(ROOT/'requirements.txt')],check=True)
ENV=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'src'])),PYTHONUNBUFFERED='1')
BUNDLES=[]
for size in ['1_5b','3b']:
    candidates=list(Path('/kaggle/input').rglob('cq_'+size+'_pilot.zip'))
    assert candidates, f'Attach the completed cq_{size}_pilot.zip output'
    by_hash={hashlib.sha256(p.read_bytes()).hexdigest():p for p in candidates}
    assert len(by_hash)==1, f'Multiple different bundles for {size}; attach only the intended run'
    BUNDLES.append(next(iter(by_hash.values())))
print('Found:',BUNDLES)
'''),
        nb.v4.new_code_cell('''OUT=WORK/'cq_comparison'
OUT.mkdir(exist_ok=True)
command=[str(PYTHON),'-m','eval.contract_quality.collect',*map(str,BUNDLES),'--out',str(OUT/'pilot_comparison.json')]
with (WORK/'cq_collection.log').open('w') as log:
    process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end='');log.write(line);log.flush()
    code=process.wait()
assert code==0, 'Collection failed; download cq_collection.log for diagnosis'
import zipfile
ARCHIVE=WORK/'cq_comparison_bundle.zip'
with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(OUT.iterdir()):archive.write(p,p.name)
from IPython.display import Markdown, FileLink, display
display(Markdown((OUT/'pilot_comparison.md').read_text()))
display(FileLink(str(ARCHIVE)))
print('COMPLETE: 320 parser records checked across four variants.')
'''),
        nb.v4.new_markdown_cell('Save cq_comparison_bundle.zip in kaggle_outputs/contract_quality/v1/ and the executed collector in notebooks/executed/contract_quality/v1/. Read pilot_comparison.md; the suite is still a synthetic draft.')
    ],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nb.validate(collector)
    for cell in collector.cells:
        if cell.cell_type=='code':compile(cell.source,'collector','exec')
    nb.write(collector,HERE/'cq_collect_results.ipynb')
    with zipfile.ZipFile(HERE/'contract_quality_pilots.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HERE.glob('cq_*.ipynb')):z.write(p,p.name)

if __name__=='__main__':build()
