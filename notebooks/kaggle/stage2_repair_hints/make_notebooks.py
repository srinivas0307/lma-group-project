"""Build the sealed repair-hints-v1 source pack and three Kaggle notebooks (gold, selective, tools).

The completed contract-check pack is reused byte for byte, the gold modules must equal the completed gold
pack, and only eval/stage2_repair_hints is new.
    .venv/bin/python -m notebooks.kaggle.stage2_repair_hints.make_notebooks
"""
import hashlib
import json
import zipfile
from pathlib import Path

import nbformat as nb

from eval.stage2_repair_hints.artifacts import ROOT, source_files, EXPERIMENT, MANIFEST

HERE = Path(__file__).resolve().parent
NAME = 'stage2_repair_hints_source'
BASELINES = {'gold': '84 correct / 0 incorrect accepted / 16 withheld (combined gold 3B)',
             'selective': '76 / 1 / 23 (contract-check selective)',
             'tools': '74 / 1 / 25 (contract-check tools)'}
ADAPTERS = {'selective': 'selective_facts_1_5b.zip (Kaggle dataset srinivasu2005/selective-3b-dataset)',
            'tools': 'tool_use_train_1_5b.zip (Kaggle dataset jagadeeshamudala/tooluse-source)'}


def frozen(zip_path, prefix, sidecar):
    with zipfile.ZipFile(zip_path) as z:
        contents = {n[len(prefix):]: z.read(n) for n in z.namelist()
                    if n.startswith(prefix) and n != prefix + 'manifest.json' and not n.endswith('/')}
        contents[sidecar] = z.read(prefix + 'manifest.json')
    return contents


def build():
    contents = frozen(ROOT / 'notebooks/kaggle/stage2_contract_check/stage2_contract_check_source.zip',
                      'stage2_contract_check_source/', '_stage2_contract_check_manifest.json')
    gold = frozen(ROOT / 'notebooks/kaggle/stage2_combined_gold/stage2_combined_gold_source.zip',
                  'stage2_combined_gold_source/', '_stage2_combined_gold_manifest.json')
    for name, data in gold.items():
        if name in contents and contents[name] != data:
            raise ValueError(f'Contract-check and gold packs disagree: {name}')
        contents.setdefault(name, data)
    for p in source_files():
        rel = str(p.relative_to(ROOT))
        if rel in contents:
            if contents[rel] != p.read_bytes():
                raise ValueError(f'Frozen source differs from the completed packs: {rel}')
        elif rel.startswith('eval/stage2_repair_hints/'):
            contents[rel] = p.read_bytes()
        else:
            raise ValueError(f'Unexpected new runtime file outside eval/stage2_repair_hints: {rel}')
    manifest = json.dumps(dict(experiment=EXPERIMENT, sha256={n: hashlib.sha256(b).hexdigest()
                                                              for n, b in sorted(contents.items())}),
                          sort_keys=True, indent=2).encode()
    sha = hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE / f'{NAME}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(f'{NAME}/manifest.json', manifest)
        for n, b in sorted(contents.items()):
            z.writestr(f'{NAME}/{n}', b)

    template = nb.read(ROOT / 'notebooks/kaggle/stage2_contract_check/stage2_contract_check_selective_kaggle.ipynb',
                       as_version=4)
    base_header = template.cells[1].source.split("METHOD='selective'")[0]
    loader = 'MANIFEST_SHA=' + repr(sha) + '\n' + template.cells[2].source.split('\n', 1)[1]
    loader = (loader.replace('stage2_contract_check_source', NAME)
              .replace("=='stage2-contract-check-3b-fp16-v1'", f"=='{EXPERIMENT}'")
              .replace("'_stage2_contract_check_manifest.json'", f"'{MANIFEST}'")
              .replace("WORK/'stage2_contract_check_packages'", "WORK/'stage2_repair_hints_packages'"))
    assert 'contract_check' not in loader, 'Loader still refers to the old pack'
    prepare_all = template.cells[3].source
    prepare_no_adapter = prepare_all.split('path_file=')[0]
    for pipeline in ('gold', 'selective', 'tools'):
        bundle = f'stage2_repair_hints_{pipeline}'
        if pipeline == 'gold':
            header = base_header + f"PIPELINE='gold'\nmatches=list(INPUT.rglob('{NAME}.zip'));print('found ZIP:' if matches else 'No intact ZIP (may be extracted):','{NAME}.zip',*matches)\n"
            prepare, adapter_arg = prepare_no_adapter + 'save_results()\n', ''
        else:
            zip_name = ADAPTERS[pipeline].split()[0]
            header = (base_header + f"PIPELINE=METHOD={pipeline!r}\n"
                      f"for name in ['{NAME}.zip','{zip_name}']:\n"
                      "    matches=list(INPUT.rglob(name));print(('found ZIP: ' if matches else 'No intact ZIP (may be extracted): ')+name,*matches)\n"
                      "print('The next cells verify source files and adapter contents, including automatically extracted datasets.')\n")
            prepare, adapter_arg = prepare_all, "'--adapter',str(ADAPTER),"
        execution = f'''DEV_COUNT=20
def execute(smoke,split='test'):
    name='{bundle}'+('_smoke' if smoke else '_dev' if split=='dev' else '')
    out=RUN_DIR/name;assert not out.exists(),'Use a fresh run folder.'
    command=[sys.executable,'-m','eval.stage2_repair_hints.runner','--pipeline',PIPELINE,{adapter_arg}'--out',str(out),'--packages',str(PACKAGES),'--split',split]
    if smoke:command+=['--smoke']
    log=RUN_DIR/(name+'.log')
    try:
        with log.open('w') as f:
            process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='');f.write(line);f.flush()
            code=process.wait()
    finally:save_results()
    assert code==0,f'Run failed: inspect {{log}} and the partial ZIP. A partial job is not a result.'
    assert json.loads((out/'completion.json').read_text())['complete']
    assert len(list((out/'tasks').glob('*.json')))==(2 if smoke else DEV_COUNT if split=='dev' else 100)
    return out.with_suffix('.zip')
SMOKE=execute(True)
DEV=execute(False,split='dev')
print('Development summary:',(DEV.with_suffix('')/'summary.json').read_text())
print('Smoke and development completed. The next cell runs 100 diagnostic test tasks freshly.')
'''
        finish = '''BUNDLE=execute(False)
from IPython.display import display,FileLink
display(FileLink(str(BUNDLE)))
print('Test summary:',(BUNDLE.with_suffix('')/'summary.json').read_text())
print('COMPLETE: download the full ZIP and this executed notebook for paired analysis (tools/analyze_repair_hints_run.py).')
'''
        inputs = (f'**{NAME}.zip** only (no parser adapter: gold contracts come from the task files)' if pipeline == 'gold'
                  else f'**{NAME}.zip** and the same adapter as before: {ADAPTERS[pipeline]}')
        intro = f'''# Stage 2 — repair hints (repair-hints-v1), {pipeline} pipeline
Paired against the completed run: {BASELINES[pipeline]}. Everything is identical (the frozen runner is executed unchanged{'' if pipeline == 'gold' else ', including the contract-check correction turn'}; same Qwen2.5-3B FP16 generator and repair model, prompts, budgets, schema guard and verifier) except the **repair agent's user message**, which adds:

* pandas recipes for the contract predicates that failed, derived only from the contract and the input columns (null cells never match a condition; keep='first'/'last' means original input order; Series.round), and
* static lints for contract-relevant pitfalls in the submitted code (sort before drop_duplicates, unrequested dropna, discarded `.cat` results, `round()`/`np.round`, overwriting a preserved column, dropping a required column, merge-based fills, groupby first()).

Code that passes verification on the first try never sees a hint. Reference programs and outputs remain evaluator-only.

Inputs: {inputs}. Enable Internet, select T4 x2, then Save Version → Run All (smoke, 20 dev, then 100 test tasks). Download **{bundle}.zip** and the executed notebook.

The hints were designed after inspecting failures on these test tasks, so test gains are diagnostic, not independent evidence.
'''
        book = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(intro)] +
                                  [nb.v4.new_code_cell(c) for c in (header, loader, prepare, execution, finish)],
                                  metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                           'name': 'python3'}, 'language_info': {'name': 'python'},
                                            'experiment': EXPERIMENT})
        nb.validate(book)
        for c in book.cells:
            if c.cell_type == 'code':
                compile(c.source, f'{pipeline}-notebook', 'exec')
        nb.write(book, HERE / f'stage2_repair_hints_{pipeline}_kaggle.ipynb')
    print(sha, len(contents))
    return sha


if __name__ == '__main__':
    build()
