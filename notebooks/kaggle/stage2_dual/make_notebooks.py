"""Small independent Colab/Kaggle launchers and CPU collector, one sealed source/data ZIP."""
import hashlib
import json
from pathlib import Path
import zipfile
import nbformat as nb
from eval.stage2_dual.artifacts import ROOT,source_files
HERE=Path(__file__).resolve().parent;COLAB=ROOT/'notebooks/colab/stage2_dual'

def build():
    paths=source_files()+list((ROOT/'benchmark/tasks').glob('*/*.yaml'))
    paths += [ROOT/'data'/name for name in ['adult_dev.csv','airbnb_dev.csv','Titanic-Dataset.csv','yellow_tripdata_2026-01.csv']]
    paths += [ROOT/'eval/stage2_independent_v2/environment.py',ROOT/'eval/stage2_independent_v2/__init__.py']
    # Routing preflight uses only open development data; no previous prediction/score files are included.
    suite=ROOT/'benchmark/contract_quality_v1';tasks=[json.loads(l) for l in (suite/'dev.jsonl').read_text().splitlines()]
    paths += [suite/'manifest.json',suite/'dev.jsonl']+[suite/name for name in {t['data_file'] for t in tasks}]
    contents={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(set(paths))}
    manifest=json.dumps(dict(experiment='stage2-dual-parser-v1',sha256={n:hashlib.sha256(b).hexdigest() for n,b in contents.items()}),sort_keys=True,indent=2).encode();sha=hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE/'stage2_dual_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('stage2_dual_source/manifest.json',manifest)
        for n,b in contents.items():z.writestr('stage2_dual_source/'+n,b)
    common_imports='''import datetime,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
'''
    loader=f'MANIFEST_SHA={sha!r}\n'+'''ROOT=WORK/('stage2_dual_source_'+MANIFEST_SHA[:12])
def unpack(raw,read):
    assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA,'Wrong source ZIP version'
    manifest=json.loads(raw);assert manifest['experiment']=='stage2-dual-parser-v1'
    for name,sha in manifest['sha256'].items():
        relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
        data=read(name);assert hashlib.sha256(data).hexdigest()==sha,f'Changed source/input: {name}'
        path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==data,f'Changed working source: {name}'
        else:path.write_bytes(data)
    sidecar=ROOT/'_stage2_dual_manifest.json'
    if sidecar.exists():assert sidecar.read_bytes()==raw,'Changed source manifest'
    else:sidecar.write_bytes(raw)
    print('Verified fresh-run source/data:',len(manifest['sha256']),'files',MANIFEST_SHA)
found=False
for path in INPUT.rglob('manifest.json'):
    if hashlib.sha256(path.read_bytes()).hexdigest()==MANIFEST_SHA:
        unpack(path.read_bytes(),lambda name:(path.parent/name).read_bytes());found=True;break
if not found:
    for path in INPUT.rglob('stage2_dual_source.zip'):
        with zipfile.ZipFile(path) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            prefix='stage2_dual_source/';name=prefix+'manifest.json'
            if name not in z.namelist():continue
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:continue
            unpack(raw,lambda name:z.read(prefix+name));found=True;break
assert found,'Attach/place stage2_dual_source.zip in the input folder.'
PACKAGES=WORK/'stage2_dual_packages'
subprocess.run([sys.executable,'-m','pip','install','-q','--upgrade','--target',str(PACKAGES),'-r',str(ROOT/'requirements.txt')],check=True)
ENV=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'src',PACKAGES])),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
'''
    def config(platform):
        if platform=='colab':
            code=common_imports+'''DRIVE_FOLDER='/content/drive/MyDrive/igt_scopefix'
from google.colab import drive
drive.mount('/content/drive')
INPUT=Path(DRIVE_FOLDER);assert INPUT.is_dir(),f'Create {INPUT} and put the required ZIPs in it.'
WORK=Path('/content/stage2_dual_work')
'''
        else:code=common_imports+"INPUT=Path('/kaggle/input')\nWORK=Path('/kaggle/working')\n"
        code+='''WORK.mkdir(parents=True,exist_ok=True)
RUN_ID=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
RUN_DIR=WORK/('run_'+RUN_ID);RUN_DIR.mkdir()
'''
        code+="DRIVE_RESULTS=INPUT/('stage2_dual_results_'+RUN_ID)\n" if platform=='colab' else 'DRIVE_RESULTS=None\n'
        code+='''def save_results():
    if DRIVE_RESULTS is not None:
        DRIVE_RESULTS.mkdir(parents=True,exist_ok=True)
        for p in RUN_DIR.iterdir():
            if p.is_file() and p.suffix in ['.zip','.log','.json','.txt']:shutil.copy2(p,DRIVE_RESULTS/p.name)
        print('Drive results:',DRIVE_RESULTS)
print('Run folder:',RUN_DIR)
'''
        return code
    prepare='''subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.20.2','peft==0.21.2','bitsandbytes==0.50.2','accelerate==1.15.0','tokenizers==0.23.3','safetensors==0.8.0','huggingface-hub==1.33.0'],check=True)
subprocess.run([sys.executable,'-m','eval.stage2_independent_v2.environment','--prepare'],cwd=ROOT,env=ENV,check=True)
subprocess.run([sys.executable,'-c',"import torch; assert torch.cuda.is_available(); assert all(torch.cuda.get_device_properties(i).total_memory/2**30>=12 for i in range(torch.cuda.device_count())); print(torch.__version__,[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"],cwd=ROOT,env=ENV,check=True)
subprocess.run([sys.executable,'-m','finetune.selective_facts.attention'],cwd=ROOT,env=ENV,check=True)
subprocess.run([sys.executable,'-m','eval.stage2_dual.preflight'],cwd=ROOT,env=ENV,check=True)
path_file=RUN_DIR/'adapter_path.txt'
subprocess.run([sys.executable,'-m','eval.stage2_dual.adapters','--method',METHOD,'--input',str(INPUT),'--work',str(WORK),'--out',str(path_file)],cwd=ROOT,env=ENV,check=True)
ADAPTER=Path(path_file.read_text());save_results()
'''
    execute='''def execute(smoke):
    name='stage2_dual_'+METHOD+'_proposed'+('_smoke' if smoke else '')
    out=RUN_DIR/name;assert not out.exists(),'Use a fresh run folder.'
    command=[sys.executable,'-m','eval.stage2_dual.runner','--method',METHOD,'--preset','proposed','--adapter',str(ADAPTER),'--out',str(out),'--packages',str(PACKAGES)]
    if smoke:command+=['--smoke']
    log=RUN_DIR/(name+'.log')
    try:
        with log.open('w') as f:
            process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='');f.write(line);f.flush()
            code=process.wait()
    finally:save_results()
    assert code==0,f'Run failed: inspect {log} and the partial ZIP. A partial job is not a result.'
    assert json.loads((out/'completion.json').read_text())['complete']
    assert len(list((out/'tasks').glob('*.json')))==(2 if smoke else 100)
    return out.with_suffix('.zip')
SMOKE=execute(True)
print('Two-task infrastructure smoke completed. Next cell runs 100 test tasks freshly.')
'''
    finish='''BUNDLE=execute(False)
from IPython.display import display,FileLink
display(FileLink(str(BUNDLE)))
if DRIVE_RESULTS is not None:print('Drive copy:',DRIVE_RESULTS/BUNDLE.name)
print('COMPLETE: download the full ZIP and this executed notebook. Run the other method independently, then the CPU collector.')
'''
    def save(folder,name,cells):
        b=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'experiment':'stage2-dual-parser-v1'})
        nb.validate(b)
        for c in b.cells:
            if c.cell_type=='code':compile(c.source,name,'exec')
        nb.write(b,folder/name)
    for platform,folder in [('colab',COLAB),('kaggle',HERE)]:
        folder.mkdir(parents=True,exist_ok=True)
        for method in ['selective','tools']:
            adapter='selective_facts_1_5b.zip' if method=='selective' else 'tool_use_train_1_5b.zip'
            intro=f'''# Stage 2 — {method} parser + normalization
Fresh raw input → intent contract → 3B code generation → verification / 7B repair → reference evaluation.

Inputs: **stage2_dual_source.zip** and **{adapter}**. This notebook is independent of the other method's outputs and all previous parser/Stage 2 result files. No training is performed. Use a GPU with at least 12 GiB memory; T4 is supported by the preflight. Only one model is resident at a time. With two GPUs the generator uses GPU 1; parser/repair use GPU 0.

Selective facts uses the corrected scope router plus operator normalization, without model correction. Tools uses the trained 1.5B tool adapter and its existing bounded loop, with normalization on contract replies. Every tool attempt/error, including placeholder queries, remains recorded. The generator, verifier, sandbox, repair limits and prompts are shared.

Run a two-task infrastructure smoke, then all 100 test tasks. A smoke can accept/reject/escalate: it checks infrastructure, not a quality threshold. The previously inspected test tasks provide diagnostic comparison, not independent research evidence. Download **stage2_dual_{method}_proposed.zip** and the executed notebook. Compare both methods using the CPU collector from this same pack.
'''
            intro+='\n**Colab:** create `MyDrive/igt_scopefix`, place the two ZIPs there, enable T4 GPU, then Run all. Use separate sessions for the two methods. Results are copied to timestamped Drive folders.\n' if platform=='colab' else '\n**Kaggle:** attach both required ZIPs as private datasets, enable Internet and T4 GPU (T4 x2 also works), then Save Version → Run All.\n'
            header=config(platform)+f'\nMETHOD={method!r}\n'+f"for name in ['stage2_dual_source.zip',{adapter!r}]:\n    matches=list(INPUT.rglob(name));print(('found   ' if matches else 'MISSING ')+name,*matches)\n"
            save(folder,f'stage2_dual_{method}_proposed_{platform}.ipynb',[nb.v4.new_markdown_cell(intro)]+[nb.v4.new_code_cell(c) for c in [header,loader,prepare,execute,finish]])
        collector='''paths=[]
for method in ['selective','tools']:
    candidates=list(INPUT.rglob('stage2_dual_'+method+'_proposed.zip'))
    by_hash={hashlib.sha256(p.read_bytes()).hexdigest():p for p in candidates}
    assert len(by_hash)==1,f'Need exactly one distinct full bundle for {method}; found {candidates}'
    paths.append(next(iter(by_hash.values())))
OUT=RUN_DIR/'stage2_dual_comparison'
subprocess.run([sys.executable,'-m','eval.stage2_dual.collect',*map(str,paths),'--out',str(OUT)],cwd=ROOT,env=ENV,check=True)
save_results()
from IPython.display import display,FileLink,Markdown
display(Markdown((OUT/'SUMMARY.md').read_text()));display(FileLink(str(OUT.with_suffix('.zip'))))
print('COMPLETE: two full Stage 2 runs verified; no models loaded.')
'''
        intro='''# Stage 2 — two-parser CPU collector
No GPU or adapter required. Provide **stage2_dual_source.zip**, **stage2_dual_selective_proposed.zip** and **stage2_dual_tools_proposed.zip**. Inputs must come from the same source pack, match software/Python versions, and cover the same complete 100 tasks. Partial/smoke/scripted or mixed-method/version bundles are refused. Results report correct acceptance, incorrect acceptance, withholding, repairs, parser costs and every parser tool failure.
'''
        intro+='\nColab: place all three ZIPs under `MyDrive/igt_scopefix` (timestamped result subfolders are searched). Run all on CPU.\n' if platform=='colab' else '\nKaggle: attach all three ZIPs as datasets and enable Internet. Run all on CPU.\n'
        save(folder,f'stage2_dual_collect_{platform}.ipynb',[nb.v4.new_markdown_cell(intro)]+[nb.v4.new_code_cell(c) for c in [config(platform),loader,collector]])
    print('Built two independent runners + collector for Colab/Kaggle; source manifest',sha,'files',len(contents))
if __name__=='__main__':build()
