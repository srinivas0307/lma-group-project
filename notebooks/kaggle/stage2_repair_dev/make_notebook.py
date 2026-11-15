"""Generate one Kaggle notebook and one source-only ZIP; no adapters or previous results."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import nbformat as nb
from eval.stage2_repair_dev.artifacts import source_files
from eval.stage2_repair_dev.suite import build_suite
HERE=Path(__file__).resolve().parent


def build():
    files={str(p.relative_to(ROOT)):p.read_bytes() for p in source_files()}
    manifest=json.dumps(dict(experiment='stage2-3b-repair-dev-v1',sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),sort_keys=True,indent=2).encode()
    sha=hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE/'stage2_repair_dev_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name,data in {'manifest.json':manifest,**files}.items():
            info=zipfile.ZipInfo('stage2_repair_dev_source/'+name,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    intro=f'''# 3B repair development — evidence, recovery and schema guard

**This is a repair-only development comparison, not the full Stage 2 evaluation.** It uses {len(build_suite())} newly constructed synthetic cases with fixed starting programs and fixture contracts. It does not load either intent parser, use an adapter, train anything, or read previous result ZIPs. References are used only for evaluation and the clearly labeled CPU infrastructure check; they are never supplied to the live model.

Four arms: **control** (original a1-v4), **evidence** (compact examples and one bounded fresh-code recovery), **schema_guard** (original agent plus declared output-column check), and **combined**. All use the same pinned **Qwen2.5-3B-Instruct in unquantized FP16**, three repair proposals and the existing token/tool limits. Recovery consumes the existing proposal budget. Successful starting programs are included as regression controls. The evidence arm also states the contract's rounding convention; its components are evaluated together in this first experiment.

Upload **stage2_repair_dev_source.zip** as one Kaggle dataset, then attach it here. Both intact ZIP and Kaggle's automatically extracted files work. Enable **Internet** and **T4 GPU** (one or two GPUs work). No 1.5B/3B adapter datasets are needed. Run all cells or Save Version → Run All. The CPU check runs first, then a two-case GPU smoke, then {len(build_suite())} × 4 = {len(build_suite())*4} paired evaluations in the full comparison.

Download `stage2_repair_dev_full.zip` and this executed notebook. The results are development diagnostics informed by previously seen failures, not an independent research benchmark. Local CPU checks cannot guarantee a particular Kaggle image or GPU service will work; the preflight checks the actual session before the full run.
'''
    config='''import datetime,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
INPUT=Path('/kaggle/input');WORK=Path('/kaggle/working')
ARMS=['control','evidence','schema_guard','combined']
RUN_DIR=WORK/('repair_dev_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
RUN_DIR.mkdir(parents=True)
print('Outputs:',RUN_DIR)
'''
    loader=f'MANIFEST_SHA={sha!r}\n'+'''ROOT=WORK/('repair_dev_source_'+MANIFEST_SHA[:12])
def unpack(raw,read):
    assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA,'Wrong source version'
    manifest=json.loads(raw)
    assert manifest['experiment']=='stage2-3b-repair-dev-v1'
    for name,expected in manifest['sha256'].items():
        relative=Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        data=read(name)
        assert hashlib.sha256(data).hexdigest()==expected,f'Changed source: {name}'
        target=ROOT/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data,f'Changed working file: {name}'
        else:target.write_bytes(data)
    (ROOT/'_repair_dev_manifest.json').write_bytes(raw)
found=False
for path in INPUT.rglob('manifest.json'):
    if hashlib.sha256(path.read_bytes()).hexdigest()==MANIFEST_SHA:
        unpack(path.read_bytes(),lambda name:(path.parent/name).read_bytes());found=True;break
if not found:
    for path in INPUT.rglob('stage2_repair_dev_source.zip'):
        with zipfile.ZipFile(path) as z:
            assert len(z.namelist())==len(set(z.namelist())),'Duplicate archive entries'
            prefix='stage2_repair_dev_source/'
            if prefix+'manifest.json' not in z.namelist():continue
            raw=z.read(prefix+'manifest.json')
            if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:continue
            unpack(raw,lambda name:z.read(prefix+name));found=True;break
assert found,'Attach the stage2_repair_dev_source.zip dataset from the same folder as this notebook.'
PACKAGES=WORK/'repair_dev_packages'
subprocess.run([sys.executable,'-m','pip','install','-q','--upgrade','--target',str(PACKAGES),'-r',str(ROOT/'requirements.txt')],check=True)
ENV=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'src',PACKAGES])),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
print('Verified source:',ROOT)
'''
    install='''subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.20.2','peft==0.21.2','bitsandbytes==0.50.2','accelerate==1.15.0','tokenizers==0.23.3','safetensors==0.8.0','huggingface-hub==1.33.0'],check=True)
subprocess.run([sys.executable,'-m','eval.stage2_independent_v2.environment','--prepare'],cwd=ROOT,env=ENV,check=True)
subprocess.run([sys.executable,'-c',"import torch; assert torch.cuda.is_available(), 'Enable a GPU'; assert torch.cuda.get_device_properties(0).total_memory/2**30>=12; print(torch.__version__,torch.cuda.get_device_name(0))"],cwd=ROOT,env=ENV,check=True)
'''
    execute='''def execute(name,extra):
    out=RUN_DIR/name;log=RUN_DIR/(name+'.log')
    command=[sys.executable,'-m','eval.stage2_repair_dev.runner','--out',str(out),'--packages',str(PACKAGES),'--arms',*ARMS,*extra]
    with log.open('w') as f:
        process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');f.write(line);f.flush()
        code=process.wait()
    assert code==0,f'Job failed; inspect {log}. Download the log and partial output ZIP for diagnosis.'
    completed=json.loads((out/'completion.json').read_text())
    assert completed['complete'] and completed['records']==completed['expected']
    return out
CPU_CHECK=execute('stage2_repair_dev_cpu_check',['--cpu-check'])
assert not json.loads((CPU_CHECK/'completion.json').read_text())['model_quality_result']
print('CPU infrastructure checks passed. These scripted scores are not model results.')
'''
    smoke="SMOKE=execute('stage2_repair_dev_smoke',['--smoke'])\nprint('GPU smoke completed. A rejected task is not an infrastructure error. The next cell runs the full comparison.')"
    full='''FULL=execute('stage2_repair_dev_full',[])
summary=json.loads((FULL/'summary.json').read_text())
assert summary['complete'] and summary['model_quality_result']
from IPython.display import display,FileLink
import pandas as pd
display(pd.DataFrame(summary['arms']).T)
display(FileLink(str(FULL.with_suffix('.zip'))))
print('Download the full ZIP and save/download this executed notebook. No separate collector is needed.')
'''
    cells=[nb.v4.new_markdown_cell(intro),nb.v4.new_code_cell(config),nb.v4.new_code_cell(loader),nb.v4.new_code_cell(install),nb.v4.new_code_cell(execute),nb.v4.new_code_cell(smoke),nb.v4.new_code_cell(full)]
    book=nb.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),experiment='stage2-3b-repair-dev-v1'))
    for index,cell in enumerate(cells):
        cell['id']=f'repair-dev-{index}'
        if cell.cell_type=='code':compile(cell.source,'repair_dev_notebook','exec')
    nb.validate(book)
    nb.write(book,HERE/'stage2_repair_dev_kaggle.ipynb')
    (HERE/'manifest_sha256.txt').write_text(sha+'\n')
    print('Built notebook and source ZIP:',sha)
if __name__=='__main__':build()
