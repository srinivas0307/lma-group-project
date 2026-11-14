"""Build Colab and Kaggle inference-only launchers with a sealed shared source ZIP."""
import hashlib
import json
from pathlib import Path
import nbformat as nb
import zipfile
from notebooks.kaggle.selective_facts.make_notebooks import files
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
COLAB=ROOT/'notebooks/colab/selective_facts_correction'

def build():
    paths=[p for p in files() if not str(p.relative_to(ROOT)).startswith(('finetune/data/','finetune/datasets/'))]
    for package in ['selective_facts_scopefix','selective_facts_correction']:
        paths+=list((ROOT/'finetune'/package).glob('*.py'))+list((ROOT/'finetune'/package).glob('*.json'))
    contents={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(set(paths))}
    manifest=json.dumps(dict(experiment='selective-facts-correction1',sha256={n:hashlib.sha256(v).hexdigest() for n,v in contents.items()}),sort_keys=True,indent=2).encode()
    sha=hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE/'selective_facts_correction_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('selective_facts_correction_source/manifest.json',manifest)
        for n,b in contents.items():z.writestr('selective_facts_correction_source/'+n,b)
    setup=f'MANIFEST_SHA={sha!r}\n'+'''MANIFEST_NAME='manifest.json'
ROOT=WORK/('selective_facts_correction_source_'+MANIFEST_SHA[:12])
def write_source(raw,read):
    assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA,'Wrong correction source ZIP version'
    manifest=json.loads(raw)
    assert manifest['experiment']=='selective-facts-correction1'
    for name,sha in manifest['sha256'].items():
        relative=Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        data=read(name);assert hashlib.sha256(data).hexdigest()==sha,f'Changed source: {name}'
        target=ROOT/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data,f'Changed working source: {name}'
        else:target.write_bytes(data)
    print('Verified correction source:',len(manifest['sha256']),'files',MANIFEST_SHA)
found=False
for path in INPUT.rglob(MANIFEST_NAME):
    if hashlib.sha256(path.read_bytes()).hexdigest()==MANIFEST_SHA:
        write_source(path.read_bytes(),lambda name:(path.parent/name).read_bytes());found=True;break
if not found:
    for path in INPUT.rglob('selective_facts_correction_source.zip'):
        with zipfile.ZipFile(path) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            prefix='selective_facts_correction_source/';name=prefix+MANIFEST_NAME
            if name not in z.namelist():continue
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:continue
            write_source(raw,lambda name:z.read(prefix+name));found=True;break
assert found,'Attach/place selective_facts_correction_source.zip in the input folder.'
PACKAGES=WORK/'correction_packages'
subprocess.run([sys.executable,'-m','pip','install','-q','--upgrade','--target',str(PACKAGES),'-r',str(ROOT/'requirements.txt')],check=True)
ENV=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'src',PACKAGES])),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
'''
    configure='''subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.20.2','peft==0.21.2','bitsandbytes==0.50.2','accelerate==1.15.0','tokenizers==0.23.3','safetensors==0.8.0','huggingface-hub==1.33.0'],check=True)
subprocess.run([sys.executable,'-m','eval.stage2_independent_v2.environment','--prepare'],cwd=ROOT,env=ENV,check=True)
# Use the host Torch/CUDA build. Record actual environment inside every full result ZIP.
subprocess.run([sys.executable,'-c',"import torch; assert torch.cuda.is_available(), 'Enable a GPU'; assert torch.cuda.get_device_properties(0).total_memory/2**30 >= 12, 'At least 12 GiB GPU memory required'; print(torch.__version__,torch.cuda.get_device_name(0))"],cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES='0'),check=True)
subprocess.run([sys.executable,'-m','finetune.selective_facts.attention'],cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES='0'),check=True)
subprocess.run([sys.executable,'-m','finetune.selective_facts_correction.preflight'],cwd=ROOT,env=ENV,check=True)
path_file=RUN_DIR/'adapter_path.txt'
subprocess.run([sys.executable,'-m','finetune.selective_facts_scopefix.adapter','--input',str(INPUT),'--work',str(WORK),'--out',str(path_file)],cwd=ROOT,env=ENV,check=True)
ADAPTER=Path(path_file.read_text());print('Verified existing epoch-2 adapter:',ADAPTER)
'''
    execute='''def save_results():
    if DRIVE_RESULTS is not None:
        DRIVE_RESULTS.mkdir(parents=True,exist_ok=True)
        for path in RUN_DIR.iterdir():
            if path.is_file() and path.suffix in ('.zip','.log','.json','.txt'):shutil.copy2(path,DRIVE_RESULTS/path.name)
        print('Copied results to Drive:',DRIVE_RESULTS)
def execute(smoke):
    out=RUN_DIR/('selective_facts_correction_smoke' if smoke else 'selective_facts_correction_eval')
    assert not out.exists(),'Use a fresh run folder; previous results are never overwritten.'
    command=[sys.executable,'-m','finetune.selective_facts_correction.evaluate','--adapter',str(ADAPTER),'--out',str(out),'--packages',str(PACKAGES)]
    if smoke:command+=['--smoke']
    log=RUN_DIR/(out.name+'.log')
    try:
        with log.open('w') as f:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES='0'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='');f.write(line);f.flush()
            code=process.wait()
    finally:save_results()
    assert code==0,f'Job failed: inspect {log} and the partial ZIP.'
    assert json.loads((out/'completion.json').read_text())['complete']
    return out
SMOKE=execute(True)
print('Three-case smoke finished. Next cell runs the full fresh paired evaluation.')
'''
    finish='''OUT=execute(False)
print((OUT/'comparison.json').read_text())
print('Download:',OUT.with_suffix('.zip'))
if DRIVE_RESULTS is not None:print('Drive copy:',DRIVE_RESULTS/'selective_facts_correction_eval.zip')
from IPython.display import display,FileLink
display(FileLink(str(OUT.with_suffix('.zip'))))
print('Download the executed notebook too. No training was performed.')
'''
    common='''import datetime,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
'''
    for platform in ['colab','kaggle']:
        if platform=='colab':
            config=common+'''# Same Drive input folder as your successfully executed scope-fix notebook.
DRIVE_FOLDER='/content/drive/MyDrive/igt_scopefix' # change only if you stored the ZIPs elsewhere
from google.colab import drive
drive.mount('/content/drive')
INPUT=Path(DRIVE_FOLDER)
assert INPUT.is_dir(),f'Create {INPUT} in Google Drive and put the two ZIP files in it.'
for name in ['selective_facts_correction_source.zip','selective_facts_1_5b.zip']:
    found=list(INPUT.rglob(name))
    print(('found   ' if found else 'MISSING ')+name,*found)
print('The correction source ZIP is new. selective_facts_scopefix_source.zip is the previous workflow and cannot replace it.')
WORK=Path('/content/selective_correction_work')
'''
        else:config=common+"INPUT=Path('/kaggle/input')\nWORK=Path('/kaggle/working')\n"
        config+='''WORK.mkdir(parents=True,exist_ok=True)
RUN_ID=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
RUN_DIR=WORK/('correction_'+RUN_ID);RUN_DIR.mkdir()
'''
        config+="DRIVE_RESULTS=INPUT/('results_'+RUN_ID)\n" if platform=='colab' else 'DRIVE_RESULTS=None\n'
        config+="print('Run folder:',RUN_DIR,'Results will be copied to:',DRIVE_RESULTS)\n"
        intro='''# Existing adapter: operator normalization and one correction
**Inference only. No retraining.** Uses your selected epoch-2 1.5B adapter and corrected selective-facts router.

Prepare two inputs: **selective_facts_correction_source.zip** (new source bundle) and your completed **selective_facts_1_5b.zip** (training output containing the adapter). No prior evaluation ZIP, original adapter or GitHub push is needed.

Three arms share one fresh initial model answer per task: **control**, **operator normalization**, and **normalization plus at most one correction**. Only whitespace around recognized operators is normalized. Schema/input-consistency checks trigger correction; gold contracts and reference outputs never enter that decision or feedback. Remaining invalid/inconsistent corrected answers are withheld, with no third call or reference-based fallback.

The 100 open-development cases are all reevaluated, including existing passes, with a three-case smoke first. Results record every raw reply, normalization, diagnostic, correction, environment and semantic score. Historical 85/100 is context; the fresh control is the paired comparison. Passing input checks does not prove the contract matches the user's intent.

Download **selective_facts_correction_eval.zip** and this executed notebook. Inspect gains, regressions and wrong-output acceptance before any Stage 2 integration.
'''
        intro+= '\n**Colab:** enable a GPU (T4 or better, at least 12 GiB), put both ZIPs in `MyDrive/igt_scopefix`, then Run all. Results are copied to a timestamped Drive folder.\n' if platform=='colab' else '\n**Kaggle:** attach both ZIPs as private datasets, enable Internet and a T4 GPU (T4 x2 also works; only GPU 0 is used), then Save Version → Run All.\n'
        book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(intro)]+[nb.v4.new_code_cell(s) for s in [config,setup,configure,execute,finish]],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'experiment':'selective-facts-correction1'})
        nb.validate(book)
        for c in book.cells:
            if c.cell_type=='code':compile(c.source,platform,'exec')
        folder=COLAB if platform=='colab' else HERE;folder.mkdir(parents=True,exist_ok=True)
        nb.write(book,folder/f'selective_facts_correction_{platform}.ipynb')
    print('Built Kaggle + Colab launchers and source ZIP',sha)
if __name__=='__main__':build()
