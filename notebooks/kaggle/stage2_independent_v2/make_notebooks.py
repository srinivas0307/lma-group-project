"""Package raw inputs and executable source: no prior result folders, model weights or Git checkout."""
import base64
import hashlib
import json
import textwrap
import zipfile
import zlib
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SYSTEMS=['baseline1','baseline2_repair','baseline3_repair','gold_contract','proposed','no_contract_to_generator']

def payload():
    files=list((ROOT/'src/igt').rglob('*.py'))
    files += [p for p in (ROOT/'eval/stage2_independent_v2').iterdir() if p.suffix in ['.py','.json']]
    files += [ROOT/'eval/equivalence.py',ROOT/'eval/stage2_clean/attention.py',ROOT/'eval/stage2_clean/__init__.py',ROOT/'eval/contract_quality/context.py',ROOT/'eval/contract_quality/__init__.py',ROOT/'requirements.txt']
    files += list((ROOT/'benchmark/tasks').glob('*/*.yaml'))
    files += [ROOT/'data'/name for name in ['adult_dev.csv','airbnb_dev.csv','Titanic-Dataset.csv','yellow_tripdata_2026-01.csv']]
    raw=json.dumps({str(p.relative_to(ROOT)):base64.b64encode(p.read_bytes()).decode() for p in sorted(files)},sort_keys=True).encode()
    return base64.b64encode(zlib.compress(raw,9)).decode(),hashlib.sha256(raw).hexdigest()

def bootstrap(encoded,sha):
    return '''import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path
WORK=Path('/kaggle/working')
WORK.mkdir(parents=True,exist_ok=True)
'''+f"PAYLOAD_SHA={sha!r}\nPAYLOAD=(\n"+'\n'.join(repr(line) for line in textwrap.wrap(encoded,100))+''')
raw=zlib.decompress(base64.b64decode(PAYLOAD))
assert hashlib.sha256(raw).hexdigest()==PAYLOAD_SHA
ROOT=WORK/('stage2_independent_v2_source_'+PAYLOAD_SHA[:12])
for name,encoded_source in json.loads(raw).items():
    source=base64.b64decode(encoded_source)
    relative=Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==source, f'Changed embedded source: {path}'
    else:path.write_bytes(source)
PACKAGES=WORK/'stage2_independent_v2_packages'
subprocess.run([sys.executable,'-m','pip','install','-q','--upgrade','--target',str(PACKAGES),'-r',str(ROOT/'requirements.txt')],check=True)
ENV=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'src',PACKAGES])),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
print('Source and raw inputs:',PAYLOAD_SHA)
'''

def save(name,cells):
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nb.validate(book)
    for c in cells:
        if c.cell_type=='code':compile(c.source,name,'exec')
    nb.write(book,HERE/name)

def build():
    encoded,sha=payload();common=bootstrap(encoded,sha)
    for index,preset in enumerate(SYSTEMS,1):
        cells=[nb.v4.new_markdown_cell(f'''# Independent Stage 2 v2 — {preset}
Raw dataset + request → fresh contract where applicable → generation → verification/repair → evaluation.
No previous parses, pilot outputs, Stage 1 thresholds, cached responses, GitHub push or other notebook outputs are needed.
Select **GPU T4 ×2**, Internet on. Only proposed/ablation need the original **1.5B adapter** attached as a private dataset.
Version 2 restores NF4 parser loading and pins the parser base revision. Quality recovery is not yet verified. Run stage2_parser_diagnostic.ipynb first. This method uses profiles for live parsers and hand-set conventional thresholds. The original 100 test tasks are diagnostic after prior failure analysis; these are not fresh confirmatory results.
Run a two-task infrastructure smoke check, then all 100 tasks in this notebook. Use a fresh Kaggle session.'''),
            nb.v4.new_code_cell(common),
            nb.v4.new_code_cell(f"PRESET={preset!r}\n"+'''subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.20.2','peft==0.21.2','bitsandbytes==0.50.2','accelerate==1.15.0','tokenizers==0.23.3','safetensors==0.8.0','huggingface-hub==1.33.0'],check=True)
# PEFT probes Kaggle's optional torchao package even for this ordinary LoRA adapter.
# Check adapter injection on a tiny CPU model before downloading the large models.
subprocess.run([sys.executable,'-m','eval.stage2_independent_v2.environment','--prepare'],cwd=ROOT,env=ENV,check=True)
ADAPTER=None
if PRESET in ['proposed','no_contract_to_generator']:
    spec=json.loads((ROOT/'eval/stage2_independent_v2/adapter.json').read_text())
    matches=[]
    for config in Path('/kaggle/input').rglob('adapter_config.json'):
        try:
            if json.loads(config.read_text()).get('base_model_name_or_path')==spec['model']:matches.append(config.parent)
        except ValueError:pass
    assert len(matches)==1, f'Attach exactly one original 1.5B adapter; found {matches}'
    ADAPTER=matches[0]
    for name,expected in spec['sha256'].items():assert hashlib.sha256((ADAPTER/name).read_bytes()).hexdigest()==expected, f'Wrong adapter: {name}'
print('System:',PRESET,'Adapter:',ADAPTER)
'''),
            nb.v4.new_code_cell('''def execute(smoke):
    name='stage2_independent_v2_'+PRESET+('_smoke' if smoke else '')
    out=WORK/name
    command=[sys.executable,'-m','eval.stage2_independent_v2.runner','--preset',PRESET,'--out',str(out),'--packages',str(PACKAGES)]
    if ADAPTER is not None:command+=['--adapter',str(ADAPTER)]
    if smoke:command+=['--limit','2']
    log=WORK/(name+'.log')
    with log.open('w') as f:
        process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');f.write(line);f.flush()
        code=process.wait()
    assert code==0, f'Run failed; download {log} and the partial ZIP. A partial job is not a result.'
    import zipfile
    with zipfile.ZipFile(out.with_suffix('.zip')) as z:assert json.loads(z.read('bundle.json'))['complete']
    assert len(list((out/'tasks').glob('*.json')))==(2 if smoke else 100)
    return out.with_suffix('.zip')
execute(smoke=True)
'''),
            nb.v4.new_code_cell('''BUNDLE=execute(smoke=False)
from IPython.display import FileLink,display
display(FileLink(str(BUNDLE)))
print('COMPLETE AND VALIDATED: 100 / 100')
'''),
            nb.v4.new_markdown_cell('''Download the full ZIP and executed notebook. Keep them in `kaggle_outputs/stage2/independent_v2/` and `notebooks/executed/stage2/independent_v2/`. Run each of the six notebooks independently; then attach their six full ZIPs to `stage2_independent_v2_collect.ipynb` (CPU).''')]
        save(f'stage2_{index:02d}_{preset}.ipynb',cells)
    cells=[nb.v4.new_markdown_cell('# Independent Stage 2 collector\nCPU, Internet on. Attach six full ZIPs produced by this same notebook pack. No model or adapter is required. This collector summarizes the new runs; it does not feed outputs into another system.'),
        nb.v4.new_code_cell(common),
        nb.v4.new_code_cell(f'SYSTEMS={SYSTEMS!r}\n'+'''paths=[]
for preset in SYSTEMS:
    candidates=list(Path('/kaggle/input').rglob('stage2_independent_v2_'+preset+'.zip'))
    by_hash={hashlib.sha256(p.read_bytes()).hexdigest():p for p in candidates}
    assert len(by_hash)==1, f'Attach exactly one distinct completed bundle for {preset}: {candidates}'
    paths.append(next(iter(by_hash.values())))
OUT=WORK/'stage2_independent_v2_summary'
command=[sys.executable,'-m','eval.stage2_independent_v2.collect',*map(str,paths),'--out',str(OUT)]
subprocess.run(command,cwd=ROOT,env=ENV,check=True)
from IPython.display import FileLink,Markdown,display
display(Markdown((OUT/'SUMMARY.md').read_text()))
display(FileLink(str(OUT.with_suffix('.zip'))))
print('COMPLETE: 600 fresh task records verified.')
''')]
    save('stage2_independent_v2_collect.ipynb',cells)
    proposed=nb.read(HERE/'stage2_05_proposed.ipynb',as_version=4)
    cells=[nb.v4.new_markdown_cell("""# Parser loading and input diagnostic (development only)
Run this first: four controlled arms compare original/profile input with NF4/FP16 loading.
Uses fresh development tasks, the original 1.5B adapter, fixed base revision and identical decoding/packages.
No Stage 2 outputs, GitHub push, 3B adapter, generator or repair model required.
Select GPU T4 x2, Internet on, attach the original 1.5B adapter, then Save & Run All in a fresh session.
Reference code is used only after each parser response for scoring; it never enters model messages.
This is a diagnosis, not a promise of improved accuracy. Download the final ZIP for review before running all six systems."""),
        nb.v4.new_code_cell(common),nb.v4.new_code_cell(proposed.cells[2].source),
        nb.v4.new_code_cell('''OUT=WORK/'stage2_parser_diagnostic'
assert not OUT.exists(), 'Start a fresh Kaggle session; do not mix arms from different runs.'
OUT.mkdir()
ARMS=['original_nf4','profile_nf4','original_fp16','profile_fp16']
for arm in ARMS:
    command=[sys.executable,'-m','eval.stage2_independent_v2.diagnostic','--arm',arm,'--adapter',str(ADAPTER),'--out',str(OUT/arm),'--packages',str(PACKAGES)]
    with (OUT/(arm+'.log')).open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');log.write(line);log.flush()
        code=process.wait()
    assert code==0, f'{arm} failed; download its log and partial ZIP for diagnosis.'
subprocess.run([sys.executable,'-m','eval.stage2_independent_v2.diagnostic','--summarize','--out',str(OUT)],cwd=ROOT,env=ENV,check=True)
from IPython.display import FileLink,Markdown,display
display(Markdown((OUT/'SUMMARY.md').read_text()))
display(FileLink(str(OUT.with_suffix('.zip'))))
print('COMPLETE: four controlled development arms')
''')]
    save('stage2_parser_diagnostic.ipynb',cells)
    with zipfile.ZipFile(HERE/'stage2_independent_v2_notebooks.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(HERE.glob('*.ipynb')):z.write(p,p.name)

if __name__=='__main__':build()
