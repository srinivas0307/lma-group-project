"""Small Kaggle notebook plus a sealed source/data dataset bundle."""
import hashlib
import json
import zipfile
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent

def files():
    paths=list((ROOT/'src/igt').rglob('*.py'))+list((ROOT/'finetune/selective_facts').glob('*.py'))
    paths+=list((ROOT/'finetune/data/selective_facts_v1').rglob('*.jsonl'))+list((ROOT/'finetune/data/selective_facts_v1').rglob('*.json'))+list((ROOT/'finetune/data/selective_facts_v1/data').glob('*.csv'))
    paths+=list((ROOT/'finetune/datasets').glob('*.csv'))
    suite=ROOT/'benchmark/contract_quality_v1'
    dev=[json.loads(line) for line in (suite/'dev.jsonl').read_text().splitlines()]
    paths += [suite/'dev.jsonl',suite/'manifest.json']+[suite/name for name in {t['data_file'] for t in dev}]
    # Only open-development data is packaged; selection and test tasks are absent.
    paths += [p for p in (ROOT/'benchmark/tasks').glob('*/*.yaml') if __import__('yaml').safe_load(p.read_text())['split']=='dev']
    paths += [ROOT/'data'/n for n in ['adult_dev.csv','airbnb_dev.csv','Titanic-Dataset.csv','yellow_tripdata_2026-01.csv']]
    paths += [ROOT/'eval/stage2_clean/attention.py',ROOT/'eval/stage2_clean/__init__.py',ROOT/'eval/stage2_independent_v2/adapter.json',ROOT/'eval/equivalence.py',ROOT/'eval/contract_quality/harness.py',ROOT/'eval/contract_quality/build_suite.py',ROOT/'eval/contract_quality/__init__.py',ROOT/'eval/stage2_independent_v2/environment.py',ROOT/'eval/stage2_independent_v2/__init__.py',ROOT/'requirements.txt']
    return sorted(set(paths))

def build():
    contents={str(p.relative_to(ROOT)):p.read_bytes() for p in files()}
    manifest=json.dumps(dict(experiment='selective-facts-sft-v1',sha256={n:hashlib.sha256(v).hexdigest() for n,v in contents.items()}),sort_keys=True,indent=2).encode()
    sha=hashlib.sha256(manifest).hexdigest()
    bundle=HERE/'selective_facts_training_source.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('selective_facts_source/selective_facts_manifest.json',manifest)
        for name,content in contents.items():archive.writestr('selective_facts_source/'+name,content)
    setup='''import hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
WORK=Path('/kaggle/working')
WORK.mkdir(parents=True,exist_ok=True)
'''+f'MANIFEST_SHA={sha!r}\n'+'''INPUT=Path('/kaggle/input')
MANIFEST_NAME='selective_facts_manifest.json'
ROOT=WORK/('selective_facts_source_'+MANIFEST_SHA[:12])

def write_source(manifest_bytes,read_source):
    assert hashlib.sha256(manifest_bytes).hexdigest()==MANIFEST_SHA, 'Wrong source/data bundle version'
    manifest=json.loads(manifest_bytes)
    assert manifest['experiment']=='selective-facts-sft-v1'
    for name,expected in manifest['sha256'].items():
        relative=Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts, 'Invalid source path'
        content=read_source(name)
        assert hashlib.sha256(content).hexdigest()==expected, f'Changed source/data: {name}'
        path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==content, f'Changed working copy: {name}'
        else:path.write_bytes(content)
    print('Verified source and data:',len(manifest['sha256']),'files',MANIFEST_SHA)

# Kaggle can expose uploaded ZIPs as files or unpacked dataset directories. Support both.
manifest_paths=[p for p in INPUT.rglob(MANIFEST_NAME) if hashlib.sha256(p.read_bytes()).hexdigest()==MANIFEST_SHA]
if manifest_paths:
    manifest_path=manifest_paths[0]
    write_source(manifest_path.read_bytes(),lambda name:(manifest_path.parent/name).read_bytes())
else:
    archives=list(INPUT.rglob('selective_facts_training_source.zip'))
    found=False
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist())==len(set(z.namelist())), 'Duplicate source ZIP members'
            name='selective_facts_source/'+MANIFEST_NAME
            if name not in z.namelist():continue
            manifest_bytes=z.read(name)
            if hashlib.sha256(manifest_bytes).hexdigest()!=MANIFEST_SHA:continue
            write_source(manifest_bytes,lambda relative:z.read('selective_facts_source/'+relative))
            found=True;break
    assert found, 'Attach the private dataset made from selective_facts_training_source.zip, in addition to the original 1.5B adapter.'
PACKAGES=WORK/'selective_facts_packages'
subprocess.run([sys.executable,'-m','pip','install','-q','--upgrade','--target',str(PACKAGES),'-r',str(ROOT/'requirements.txt')],check=True)
ENV=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[ROOT,ROOT/'src',PACKAGES])),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
'''
    configure='''subprocess.run([sys.executable,'-m','pip','install','-q','transformers==5.20.2','peft==0.21.2','bitsandbytes==0.50.2','accelerate==1.15.0','tokenizers==0.23.3','safetensors==0.8.0','huggingface-hub==1.33.0'],check=True)
subprocess.run([sys.executable,'-m','eval.stage2_independent_v2.environment','--prepare'],cwd=ROOT,env=ENV,check=True)
subprocess.run([sys.executable,'-m','finetune.selective_facts.preflight'],cwd=ROOT,env=ENV,check=True)
subprocess.run([sys.executable,'-c',"import torch; assert torch.__version__=='2.11.0+cu130'; assert torch.cuda.device_count()==2; assert all('T4' in torch.cuda.get_device_name(i) for i in (0,1)); print('T4 x2 preflight passed')"],cwd=ROOT,env=ENV,check=True)
spec=json.loads((ROOT/'eval/stage2_independent_v2/adapter.json').read_text())
matches=[]
for config in Path('/kaggle/input').rglob('adapter_config.json'):
    try:
        if json.loads(config.read_text()).get('base_model_name_or_path')==spec['model']:
            if all((config.parent/n).is_file() and hashlib.sha256((config.parent/n).read_bytes()).hexdigest()==sha for n,sha in spec['sha256'].items()):matches.append(config.parent)
    except (ValueError,OSError):pass
assert len(matches)==1, f'Attach exactly one original 1.5B adapter dataset; found {matches}'
BASELINE_ADAPTER=matches[0]
print('Original control adapter:',BASELINE_ADAPTER)
# Test the same single-visible-GPU topology used by training and semantic evaluation.
for physical_gpu in ['0','1']:
    subprocess.run([sys.executable,'-m','finetune.selective_facts.attention'],cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES=physical_gpu),check=True)
'''
    execute='''def execute(smoke):
    out=WORK/('selective_facts_1_5b_smoke' if smoke else 'selective_facts_1_5b')
    assert not out.exists(), 'Start a fresh session or choose a new output directory.'
    command=[sys.executable,'-m','finetune.selective_facts.train','--data',str(ROOT/'finetune/data/selective_facts_v1'),'--out',str(out),'--baseline-adapter',str(BASELINE_ADAPTER),'--packages',str(PACKAGES)]
    if smoke:command+=['--smoke']
    # Training sees only physical GPU 0. Semantic subprocesses explicitly use physical GPU 1.
    training_env=dict(ENV,CUDA_VISIBLE_DEVICES='0')
    log=WORK/(out.name+'.log')
    with log.open('w') as f:
        process=subprocess.Popen(command,cwd=ROOT,env=training_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');f.write(line);f.flush()
        code=process.wait()
    assert code==0, f'Job failed. Download {log} and the partial ZIP for diagnosis.'
    assert json.loads((out/'completion.json').read_text())['complete']
    return out
SMOKE=execute(True)
print('Infrastructure smoke completed. Starting full fresh training in the next cell.')
'''
    finish='''OUT=execute(False)
selection=json.loads((OUT/'selection.json').read_text())
print('Passed development screening:',selection['eligible'])
print('Original baseline:',json.dumps(selection['baseline'],indent=2))
print('New selected candidate:',json.dumps(selection['selected'],indent=2))
from IPython.display import FileLink,display
display(FileLink(str(OUT.with_suffix('.zip'))))
print('COMPLETE: training, full validation loss and semantic checkpoint evaluation.')
if not selection['eligible']:print('No checkpoint met the declared screening rule. Share the results for analysis before Stage 2.')
'''
    cells=[nb.v4.new_markdown_cell('''# Selective facts parser — train and evaluate
Use **GPU T4 x2**, **Internet on**, attach **two private datasets**: the contents of `selective_facts_training_source.zip`, plus your original **1.5B adapter** for the control comparison. No 3B adapter, GitHub push or earlier result ZIP is needed.

The job trains a **new** adapter from the pinned 1.5B base. It uses selective facts for supported statistic-fill requests, with the request last. Replay examples and new scoped examples use this same input format.

A two-step infrastructure smoke job runs first, followed by a fresh full two-epoch job. Each epoch is scored by free generation on 80 synthetic open-development tasks and 20 existing development tasks. The original adapter uses its original input format as the control. These scores are diagnostic; synthetic additions are automatically checked drafts without independent review. Training validation sources are separate from training sources. No held-out selection inference or test scoring is run.

Download `selective_facts_1_5b.zip` and the executed notebook. A valid completed run may still fail quality screening. Inspect its result before using the adapter in Stage 2.'''),nb.v4.new_code_cell(setup),nb.v4.new_code_cell(configure),nb.v4.new_code_cell(execute),nb.v4.new_code_cell(finish)]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13'},'experiment':'selective-facts-sft-v1'})
    nb.validate(book)
    for cell in cells:
        if cell.cell_type=='code':compile(cell.source,'selective_facts_train.ipynb','exec')
    nb.write(book,HERE/'selective_facts_train.ipynb')
    with zipfile.ZipFile(HERE/'selective_facts_training_notebook.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(HERE/'selective_facts_train.ipynb','selective_facts_train.ipynb')
    assert (HERE/'selective_facts_train.ipynb').stat().st_size<100_000, 'Keep the Kaggle commit payload small'
    print('Wrote',HERE/'selective_facts_train.ipynb')
    print('Upload as private Kaggle dataset:',bundle)
if __name__=='__main__':build()
