"""Seal the new configuration alongside the exact completed 3B-gold source."""
import hashlib,json,zipfile
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent

def build():
    original=ROOT/'notebooks/kaggle/stage2_combined_gold'
    with zipfile.ZipFile(original/'stage2_combined_gold_source.zip') as z:
        prefix='stage2_combined_gold_source/'
        contents={n[len(prefix):]:z.read(n) for n in z.namelist() if n.startswith(prefix) and n!=prefix+'manifest.json'}
        contents['_stage2_combined_gold_manifest.json']=z.read(prefix+'manifest.json')
    for p in (ROOT/'eval/stage2_combined_gold_7b').glob('*.py'):contents[str(p.relative_to(ROOT))]=p.read_bytes()
    contents['tools/collect_stage2_combined_gold.py']=(ROOT/'tools/collect_stage2_combined_gold.py').read_bytes()
    manifest=json.dumps(dict(experiment='stage2-combined-gold-7b-nf4-v1',sha256={n:hashlib.sha256(b).hexdigest() for n,b in sorted(contents.items())}),sort_keys=True,indent=2).encode()
    sha=hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE/'stage2_combined_gold_7b_source.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('stage2_combined_gold_7b_source/manifest.json',manifest)
        for n,b in contents.items():z.writestr('stage2_combined_gold_7b_source/'+n,b)
    old=nb.read(original/'stage2_combined_gold_3b_kaggle.ipynb',as_version=4)
    cells=[old.cells[i].source.replace('stage2_combined_gold','stage2_combined_gold_7b').replace('stage2_combined_gold_7b_3b','stage2_combined_gold_7b').replace('stage2-combined-gold-3b-fp16-v1','stage2-combined-gold-7b-nf4-v1') for i in range(1,6)]
    cells[1]='MANIFEST_SHA='+repr(sha)+'\n'+cells[1].split('\n',1)[1]
    cells[2]=cells[2].replace('eval.stage2_combined_3b.preflight','eval.stage2_combined_gold_7b.preflight')
    cells[3]=cells[3].replace("SMOKE=execute(True)","print('STAGE 1/3: two infrastructure smoke tasks')\nSMOKE=execute(True)").replace("DEV=execute(False,split='dev')","print('STAGE 2/3: 20 development tasks; the next cell runs 100 test tasks')\nDEV=execute(False,split='dev')")
    cells[4]=cells[4].replace('BUNDLE=execute(False)',"print('STAGE 3/3: full 100-task diagnostic evaluation')\nBUNDLE=execute(False)").replace('Predicted-contract outputs are optional in the next cell.','The completed 3B gold bundle is optional in the next cell.')
    optional="""# Optional CPU comparison with the completed 3B FP16 gold run (84/0/16).
# Attach stage2_combined_gold_3b.zip as a separate dataset. No previous output is used for inference.
candidates=list(INPUT.rglob('stage2_combined_gold_3b.zip'))
for path in INPUT.rglob('plan.json'):
    try:
        plan=json.loads(path.read_text())
        if plan.get('experiment')!='stage2-combined-gold-3b-fp16-v1' or plan.get('smoke') or plan.get('split')!='test':continue
        folder=path.parent;seal=json.loads((folder/'bundle.json').read_text())
    except (OSError,ValueError):continue
    assert seal['complete'],'Incomplete control bundle'
    local=RUN_DIR/('repacked_control_'+str(len(candidates))+'.zip')
    with zipfile.ZipFile(local,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,digest in seal['sha256'].items():
            relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
            data=(folder/relative).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
            archive.writestr(name,data)
        archive.writestr('bundle.json',json.dumps(seal))
    candidates.append(local)
distinct={}
for path in candidates:
    with zipfile.ZipFile(path) as z:seal=json.loads(z.read('bundle.json'))
    distinct[json.dumps(seal,sort_keys=True)]=path
assert len(distinct)<=1,'Multiple different complete 3B controls attached'
if distinct:
    OUT=RUN_DIR/'stage2_combined_gold_7b_comparison'
    subprocess.run([sys.executable,'-m','eval.stage2_combined_gold_7b.collect','--gold',str(BUNDLE),'--baseline',str(next(iter(distinct.values()))),'--out',str(OUT)],cwd=ROOT,env=ENV,check=True)
    display(Markdown((OUT/'SUMMARY.md').read_text()));display(FileLink(str(OUT.with_suffix('.zip'))))
else:
    print('No 3B control attached. The 7B run and summary are complete; download them for local comparison.')
"""
    intro='''# Combined gold contracts — 7B NF4 repair
**Generator: pinned Qwen2.5-3B-Instruct, unquantized FP16. Repair: pinned Qwen2.5-7B-Instruct, NF4 with float16 computation.** Uses the same gold contracts, combined evidence/recovery loop, schema guard, prompts, seed and budgets as the completed combined 3B gold run (84 correct, zero incorrect accepted, 16 withheld). No intent parser, adapter or training. One model resident at a time; generator on GPU 1, repair on GPU 0 with T4 x2.

1. Upload **stage2_combined_gold_7b_source.zip** as a private Kaggle dataset and attach it. Intact ZIP and automatically extracted source are supported.
2. Enable **Internet**, choose **T4 x2**, then **Save Version → Run All**.
3. The notebook runs **2 smoke tasks → 20 development tasks → 100 evaluation tasks**. The 20-task development stage is followed by a separate full-evaluation cell. Each uses a fresh output directory.
4. Download **stage2_combined_gold_7b.zip**, **stage2_combined_gold_7b_summary.zip** and the executed notebook. Partial outputs are not final results.

No GitHub push or previous results are needed. Optional: attach the completed **stage2_combined_gold_3b.zip** as a separate dataset for the last CPU comparison cell; it is never used during inference. The new collector fixes the numeric YAML mapping-key versus JSON representation issue and retains strict validation. The control's original files remain unchanged.

This changes both repair model size and precision. It is a repair-configuration comparison, not a pure model-size ablation. Previously inspected tasks are diagnostic. Fresh initial generation may differ and the optional comparison reports initial-code agreement. A mismatch in software/GPU environment or other settings stops only comparison; completed inference ZIPs remain available. CPU checks cannot guarantee GPU execution or improved model quality.
'''
    b=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(intro)]+[nb.v4.new_code_cell(c) for c in cells+[optional]],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'experiment':'stage2-combined-gold-7b-nf4-v1'})
    nb.validate(b)
    for c in b.cells:
        if c.cell_type=='code':compile(c.source,'gold-7b-notebook','exec')
    nb.write(b,HERE/'stage2_combined_gold_7b_kaggle.ipynb')
    print('Built',len(contents),'sealed files; manifest',sha)
if __name__=='__main__':build()
