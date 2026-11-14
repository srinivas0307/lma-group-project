"""Small inference-only Kaggle launcher plus sealed runtime/open-development data."""
import hashlib
import json
from pathlib import Path
import re
import zipfile
import nbformat as nb
from notebooks.kaggle.selective_facts.make_notebooks import files
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

def build():
    paths=[p for p in files() if not str(p.relative_to(ROOT)).startswith(('finetune/data/','finetune/datasets/'))]
    paths+=list((ROOT/'finetune/selective_facts_scopefix').glob('*.py'))+list((ROOT/'finetune/selective_facts_scopefix').glob('*.json'))
    contents={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(set(paths))}
    manifest=json.dumps(dict(experiment='selective-facts-scopefix1',sha256={n:hashlib.sha256(v).hexdigest() for n,v in contents.items()}),sort_keys=True,indent=2).encode()
    sha=hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE/'selective_facts_scopefix_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('selective_facts_scopefix_source/selective_facts_scopefix_manifest.json',manifest)
        for name,data in contents.items():z.writestr('selective_facts_scopefix_source/'+name,data)
    old=nb.read(ROOT/'notebooks/kaggle/selective_facts/selective_facts_train.ipynb',as_version=4)
    setup=re.sub(r"MANIFEST_SHA='[^']+'",f'MANIFEST_SHA={sha!r}',old.cells[1].source)
    setup=setup.replace('selective_facts_training_source.zip','selective_facts_scopefix_source.zip').replace('selective_facts_manifest.json','selective_facts_scopefix_manifest.json').replace("'selective_facts_source_", "'selective_facts_scopefix_source_").replace("'selective_facts_source/", "'selective_facts_scopefix_source/").replace('selective-facts-sft-v1','selective-facts-scopefix1').replace('selective_facts_packages','selective_facts_scopefix_packages').replace('in addition to the original 1.5B adapter.','and the training output selective_facts_1_5b.zip.')
    configure=old.cells[2].source.split('spec=json.loads')[0]
    configure='\n'.join(line for line in configure.splitlines() if "'finetune.selective_facts.preflight'" not in line)+'\n'
    configure+='''# No optimizer or training is run. Check the one-visible-GPU inference topology.
subprocess.run([sys.executable,'-m','finetune.selective_facts.attention'],cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES='0'),check=True)
subprocess.run([sys.executable,'-m','finetune.selective_facts_scopefix.preflight'],cwd=ROOT,env=ENV,check=True)
ADAPTER_PATH_FILE=WORK/'scopefix_adapter_path.txt'
subprocess.run([sys.executable,'-m','finetune.selective_facts_scopefix.adapter','--out',str(ADAPTER_PATH_FILE)],cwd=ROOT,env=ENV,check=True)
ADAPTER=Path(ADAPTER_PATH_FILE.read_text())
print('Existing epoch-2 adapter:',ADAPTER)
'''
    execute='''def execute(smoke):
    out=WORK/('selective_facts_scopefix_smoke' if smoke else 'selective_facts_scopefix_eval')
    assert not out.exists(), 'Start a fresh session: existing results are never overwritten.'
    command=[sys.executable,'-m','finetune.selective_facts_scopefix.evaluate','--adapter',str(ADAPTER),'--out',str(out),'--packages',str(PACKAGES)]
    if smoke:command+=['--smoke']
    log=WORK/(out.name+'.log')
    with log.open('w') as f:
        process=subprocess.Popen(command,cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES='0'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');f.write(line);f.flush()
        code=process.wait()
    assert code==0, f'Job failed. Download {log} and the partial ZIP for diagnosis.'
    assert json.loads((out/'completion.json').read_text())['complete']
    return out
SMOKE=execute(True)
print('Two-task inference smoke completed. The next cell evaluates all 100 development cases.')
'''
    finish='''OUT=execute(False)
comparison=json.loads((OUT/'comparison.json').read_text())
print(json.dumps(comparison,indent=2))
from IPython.display import display,FileLink
display(FileLink(str(OUT.with_suffix('.zip'))))
print('Download selective_facts_scopefix_eval.zip and this executed notebook for analysis. No training was run.')
'''
    intro='''# Selective facts — corrected scope, existing weights
This is an **inference-only parser evaluation**. It reuses your selected full epoch-2 1.5B adapter. It does not train a model or run the six Stage 2 systems.

Choose **GPU T4 x2**, **Internet on**, and attach two private datasets:
1. The contents of **selective_facts_scopefix_source.zip** supplied with this notebook.
2. Your completed training output **selective_facts_1_5b.zip** (not the smoke ZIP). Kaggle can expose ZIPs as files or unpacked directories; both are supported.

No original adapter dataset, GitHub push, manual adapter path or earlier Stage 2 results are required. The loader verifies the exact selected weights and original SF1 metadata without changing them.

The scope correction prevents “Ignore missing values when computing statistics” from being interpreted as a row condition. It preserves the original user request and facts format. Preflight verifies that 12 previously blocked development cases now receive facts and the other 88 prompts remain exactly unchanged.

After a two-task inference smoke, it evaluates all 100 open-development cases freshly. `comparison.json` compares them with the historical original-adapter and SF1 results, with per-task gains/losses and the 12 blocked cases listed separately. Historical scores are comparisons, not reused current predictions. No selection/test inference is run. These development scores are not an independent benchmark or end-to-end Stage 2 accuracy.

Download **selective_facts_scopefix_eval.zip** and the executed notebook. Improved scores are not guaranteed; inspect the result before full Stage 2 runs.
'''
    book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(intro),nb.v4.new_code_cell(setup),nb.v4.new_code_cell(configure),nb.v4.new_code_cell(execute),nb.v4.new_code_cell(finish)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13'},'experiment':'selective-facts-scopefix1'})
    nb.validate(book)
    for c in book.cells:
        if c.cell_type=='code':compile(c.source,'selective_facts_scopefix_eval.ipynb','exec')
    nb.write(book,HERE/'selective_facts_scopefix_eval.ipynb')
    print('Built scopefix notebook and source ZIP:',sha,'files',len(contents))
if __name__=='__main__':build()
