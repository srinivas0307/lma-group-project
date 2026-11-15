"""Build a new sealed gold experiment without modifying the completed combined pack."""
import hashlib,json,zipfile
from pathlib import Path
import nbformat as nb
from eval.stage2_combined_gold.artifacts import ROOT
HERE=Path(__file__).resolve().parent

def build():
    original=ROOT/'notebooks/kaggle/stage2_combined_3b'
    with zipfile.ZipFile(original/'stage2_combined_3b_source.zip') as z:
        prefix='stage2_combined_3b_source/'
        contents={n[len(prefix):]:z.read(n) for n in z.namelist() if n.startswith(prefix) and n!=prefix+'manifest.json'}
        contents['_stage2_combined_3b_manifest.json']=z.read(prefix+'manifest.json')
    # Preserve every old byte: package only new modules alongside that frozen source.
    for p in (ROOT/'eval/stage2_combined_gold').glob('*.py'):contents[str(p.relative_to(ROOT))]=p.read_bytes()
    manifest=json.dumps(dict(experiment='stage2-combined-gold-3b-fp16-v1',sha256={n:hashlib.sha256(b).hexdigest() for n,b in sorted(contents.items())}),sort_keys=True,indent=2).encode()
    sha=hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE/'stage2_combined_gold_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('stage2_combined_gold_source/manifest.json',manifest)
        for n,b in contents.items():z.writestr('stage2_combined_gold_source/'+n,b)
    old=nb.read(original/'stage2_combined_3b_selective_proposed_kaggle.ipynb',as_version=4)
    header=old.cells[1].source.split("METHOD='selective'")[0]
    header=header.replace('stage2_combined_3b_results','stage2_combined_gold_results')
    loader=old.cells[2].source
    loader='MANIFEST_SHA='+repr(sha)+'\n'+loader.split('\n',1)[1]
    loader=loader.replace('stage2_combined_3b','stage2_combined_gold').replace('stage2-combined-3b-fp16-v1','stage2-combined-gold-3b-fp16-v1')
    prepare=old.cells[3].source.split('path_file=')[0]
    # Original preflight remains useful, but does not load an adapter or model.
    execution=old.cells[4].source
    execution=execution.replace("'stage2_combined_3b_'+METHOD+'_proposed'","'stage2_combined_gold_3b'")
    execution=execution.replace("'eval.stage2_combined_3b.runner','--method',METHOD,'--preset','proposed','--adapter',str(ADAPTER)","'eval.stage2_combined_gold.runner','--method','gold','--preset','gold_contract'")
    finish="""BUNDLE=execute(False)
OUT=RUN_DIR/'stage2_combined_gold_summary'
subprocess.run([sys.executable,'-m','eval.stage2_combined_gold.collect','--gold',str(BUNDLE),'--out',str(OUT)],cwd=ROOT,env=ENV,check=True)
from IPython.display import display,FileLink,Markdown
display(Markdown((OUT/'SUMMARY.md').read_text()))
display(FileLink(str(BUNDLE)));display(FileLink(str(OUT.with_suffix('.zip'))))
print('Download the gold bundle and this executed notebook. Predicted-contract outputs are optional in the next cell.')
"""
    comparison="""# Optional: attach the two COMPLETED combined result bundles to compare task by task.
# This cell skips comparison when neither result is attached. The gold run is independent.
paths=[]
for method in ['selective','tools']:
    name='stage2_combined_3b_'+method+'_proposed.zip'
    candidates=list(INPUT.rglob(name))
    # Support Kaggle auto-extracted datasets as well as intact ZIPs.
    for path in INPUT.rglob('plan.json'):
        try:
            plan=json.loads(path.read_text())
            if plan.get('experiment')!='stage2-combined-3b-fp16-v1' or plan.get('method')!=method or plan.get('smoke') or plan.get('split')!='test':continue
            folder=path.parent;seal=json.loads((folder/'bundle.json').read_text())
        except (OSError,ValueError):continue
        assert seal['complete'],'Incomplete predicted run'
        local=RUN_DIR/('predicted_'+method+'_'+str(len(candidates))+'.zip')
        with zipfile.ZipFile(local,'w',zipfile.ZIP_DEFLATED) as z:
            for name,digest in seal['sha256'].items():
                relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
                data=(folder/relative).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
                z.writestr(name,data)
            z.writestr('bundle.json',json.dumps(seal))
        candidates.append(local)
    distinct={}
    for path in candidates:
        with zipfile.ZipFile(path) as z:seal=json.loads(z.read('bundle.json'))
        distinct[json.dumps(seal,sort_keys=True)]=path
    assert len(distinct)<=1,f'Multiple different full results for {method}'
    paths.extend(distinct.values())
if paths:
    OUT=RUN_DIR/'stage2_combined_gold_comparison'
    subprocess.run([sys.executable,'-m','eval.stage2_combined_gold.collect','--gold',str(BUNDLE),'--predicted',*map(str,paths),'--out',str(OUT)],cwd=ROOT,env=ENV,check=True)
    display(Markdown((OUT/'SUMMARY.md').read_text()));display(FileLink(str(OUT.with_suffix('.zip'))))
else:
    print('No predicted bundles attached. Gold results are complete; upload them for local comparison.')
"""
    intro='''# Combined 3B — gold-contract diagnostic
Uses the same pinned Qwen2.5-3B-Instruct FP16 generator and combined repair loop as the completed selective-facts and tool runs. Supplies each task's gold contract to BOTH generation and verification. No intent-parser model, adapter, training or old result is required. Parser calls/tokens are zero. Reference programs remain evaluator-only.

Attach **stage2_combined_gold_source.zip**, enable **Internet**, select **T4 x2**, then **Save Version → Run All**. Run two smoke tasks, 20 fresh development tasks and 100 fresh diagnostic test tasks, each in separate output folders. Smoke acceptance is not a quality threshold. Download **stage2_combined_gold_3b.zip**, the summary ZIP and the executed notebook. Partial bundles are not completed results.

Optional: attach the completed `stage2_combined_3b_selective_proposed.zip` and `stage2_combined_3b_tools_proposed.zip` as SEPARATE datasets for a strict paired comparison in the final cell. Automatically extracted datasets are supported. They are not used during inference. Environment/settings mismatches stop comparison, while the completed gold bundle remains available. You can instead send back the gold bundle for local analysis.

This measures the combined effect of gold contracts on generation and checking/repair. Initial code can differ from predicted-contract runs. Gold is an oracle diagnostic, not a deployable system or a guaranteed upper bound; previously inspected tasks are not fresh held-out evidence. Total cost excludes parser inference in this run; generation/repair costs are separately reported.
'''
    book=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(intro)]+[nb.v4.new_code_cell(c) for c in [header,loader,prepare,execution,finish,comparison]],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'experiment':'stage2-combined-gold-3b-fp16-v1'})
    nb.validate(book)
    for c in book.cells:
        if c.cell_type=='code':compile(c.source,'gold-notebook','exec')
    nb.write(book,HERE/'stage2_combined_gold_3b_kaggle.ipynb')
    print(sha,len(contents))
if __name__=='__main__':build()
