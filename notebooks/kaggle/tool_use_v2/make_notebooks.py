"""Build small Kaggle launchers sharing one sealed code/data dataset."""
import hashlib
import json
from pathlib import Path
import zipfile
import nbformat as nb
from notebooks.kaggle.selective_facts.make_notebooks import files
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

def build():
    paths=files()+list((ROOT/'finetune/tool_use').glob('*.py'))+list((ROOT/'eval/stage2_tools').glob('*.py'))+list((ROOT/'finetune/data/tool_use_v2').glob('*'))
    contents={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(set(paths)) if p.is_file()}
    manifest=json.dumps(dict(experiment='tool-use-v2',sha256={n:hashlib.sha256(v).hexdigest() for n,v in contents.items()}),sort_keys=True,indent=2).encode()
    sha=hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE/'tool_use_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('tool_use_source/tool_use_manifest.json',manifest)
        for name,data in contents.items():z.writestr('tool_use_source/'+name,data)
    old=nb.read(ROOT/'notebooks/kaggle/selective_facts/selective_facts_train.ipynb',as_version=4)
    setup=old.cells[1].source
    import re
    setup=re.sub(r"MANIFEST_SHA='[^']+'",f'MANIFEST_SHA={sha!r}',setup)
    setup=setup.replace('selective_facts_training_source.zip','tool_use_source.zip').replace('selective_facts_manifest.json','tool_use_manifest.json').replace('selective_facts_source','tool_use_source').replace('selective-facts-sft-v1','tool-use-v2').replace('selective_facts_packages','tool_use_packages').replace(', in addition to the original 1.5B adapter.','. No original adapter is required.')
    preflight=old.cells[2].source.split('spec=json.loads')[0]
    preflight=preflight.replace("'finetune.selective_facts.preflight'","'finetune.tool_use.preflight','--size',MODEL_SIZE")
    preflight+='''for physical_gpu in ['0','1']:
    subprocess.run([sys.executable,'-m','finetune.selective_facts.attention'],cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES=physical_gpu),check=True)
'''
    for mode,title in [('base','Direct base-model tools — no adapter'),('train','Train a new tool-use adapter'),('adapter','Evaluate a trained tool-use adapter')]:
        config='''MODEL_SIZE = "1_5b"  # choose "1_5b" or "3b"; use the same size for a fair comparison
assert MODEL_SIZE in ("1_5b", "3b")
'''
        if mode=='adapter':config+='''ADAPTER = "/kaggle/input/your-tool-adapter/adapter"  # folder containing tool_protocol.json
'''
        module='train' if mode=='train' else 'evaluate'
        execute=f'''def execute(smoke):
    out=WORK/('tool_use_{mode}_'+MODEL_SIZE+('_smoke' if smoke else ''))
    assert not out.exists(), 'Use a fresh session: existing results will not be overwritten.'
    command=[sys.executable,'-m','finetune.tool_use.{module}','--size',MODEL_SIZE,'--out',str(out),'--packages',str(PACKAGES)]
'''
        if mode=='train':execute+="    command+=['--data',str(ROOT/'finetune/data/tool_use_v2')]\n"
        if mode=='adapter':execute+="    command+=['--adapter',ADAPTER]\n"
        execute+='''    if smoke:command+=['--smoke']
    log=WORK/(out.name+'.log')
    with log.open('w') as f:
        process=subprocess.Popen(command,cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES='0'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='');f.write(line);f.flush()
        code=process.wait()
    assert code==0, f'Job failed. Download {log} and the partial ZIP for diagnosis.'
    return out
SMOKE=execute(True)
print('Smoke passed. The next cell starts a fresh full job.')
'''
        finish='''OUT=execute(False)
from IPython.display import display,FileLink
display(FileLink(str(OUT.with_suffix('.zip'))))
'''
        finish+="print((OUT/'selection.json').read_text())\n" if mode=='train' else "print((OUT/'summary.json').read_text())\n"
        intro=f'''# {title}
Choose **GPU T4 x2**, enable **Internet**, and attach the private dataset made from **tool_use_source.zip**. Set MODEL_SIZE below. No old contract adapter, selective-facts run, GitHub push or earlier result bundle is required.

Both arms use NF4, the same prompt, JSON tool protocol, bounded tool loop and open-development tasks. Calls are chosen by the model, not automatically inserted. This is custom JSON tool use, not native function calling.

This notebook evaluates the **intent parser**, not the six end-to-end Stage 2 systems. It scores 80 synthetic and 20 existing development tasks and records actual tool calls, correct-reference acceptance and wrong-output acceptance. These are development diagnostics, not an independent test benchmark.

The training notebook first runs a two-step smoke job, then fresh two-epoch QLoRA training. It evaluates the unadapted base and both epoch checkpoints and selects the best candidate under the documented semantic screening rule. A completed training job does not guarantee an eligible adapter. Only the adapter evaluation notebook requires the NEW tool adapter dataset.

Download the output ZIP and executed notebook. Do not run both sizes in the same output directory or attach an old contract/selective-facts adapter here.
'''
        book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(intro),nb.v4.new_code_cell(config),nb.v4.new_code_cell(setup),nb.v4.new_code_cell(preflight),nb.v4.new_code_cell(execute),nb.v4.new_code_cell(finish)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
        nb.validate(book)
        for cell in book.cells:
            if cell.cell_type=='code':compile(cell.source,mode,'exec')
        nb.write(book,HERE/f'tool_use_{mode}.ipynb')
    print('Built three small launchers and tool_use_source.zip',sha)
if __name__=='__main__':build()
