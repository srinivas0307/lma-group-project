"""Build the sealed baselines-hints source pack and four Kaggle notebooks.

The completed repair-hints pack (which itself contains the contract-check, combined and gold packs) is reused byte
for byte; only eval/stage2_baselines_hints is new.
    .venv/bin/python -m notebooks.kaggle.stage2_baselines_hints.make_notebooks
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import nbformat as nb

sys.path[:0] = [str(Path(__file__).resolve().parents[3] / 'src')]

from eval.stage2_baselines_hints.artifacts import ROOT, source_files, EXPERIMENT, MANIFEST
from eval.stage2_baselines_hints.config import RUNS

HERE = Path(__file__).resolve().parent
NAME = 'stage2_baselines_hints_source'
HINTS = 'stage2_repair_hints_source'
ADAPTERS = {'selective': 'selective_facts_1_5b.zip (Kaggle dataset srinivasu2005/selective-3b-dataset)',
            'tools': 'tool_use_train_1_5b.zip (Kaggle dataset jagadeeshamudala/tooluse-source)'}
TITLES = {
    'baseline2': ('Baseline 2 — conventional checks with repair (no contract)',
                  'No parser and no contract. The 3B generator gets the request only. Candidates are checked by the '
                  'conventional data-quality checks with the same dev-tuned thresholds as the reported six-system run. '
                  'Repair uses the improved 3B loop (failure evidence, one fresh recovery) without its contract-derived '
                  'parts: no contract text, no schema guard, no hints.',
                  'six-system clean_v1 baseline 2 (7B NF4 repair): 72% final accuracy'),
    'baseline3': ('Baseline 3 — prompted (not fine-tuned) parser',
                  'The parser is the base Qwen2.5-3B model with the zero-shot p1-v3 prompt (no adapter, no selective '
                  'facts, no tools), followed by the same contract-check correction turn. As in the six-system plan, the '
                  'generator gets the request only; the contract drives checking and repair (with repair hints).',
                  'no_contract_selective / no_contract_tools from this experiment (same pipeline, fine-tuned parsers)'),
    'no_contract_selective': ('No contract to generator — selective-facts parser',
                              'Fine-tuned 1.5B selective-facts parser and contract check, exactly as in the proposed run. '
                              'The only change from proposed: the generator gets the request only. The contract still '
                              'drives checking and repair (with repair hints).',
                              'proposed selective with repair hints: 82 / 2 / 16'),
    'no_contract_tools': ('No contract to generator — tool-use parser',
                          'Fine-tuned 1.5B tool-use parser and contract check, exactly as in the proposed run. The only '
                          'change from proposed: the generator gets the request only. The contract still drives checking '
                          'and repair (with repair hints).',
                          'proposed tools with repair hints: 80 / 1 / 19'),
}


def frozen(zip_path, prefix, sidecar):
    with zipfile.ZipFile(zip_path) as z:
        contents = {n[len(prefix):]: z.read(n) for n in z.namelist()
                    if n.startswith(prefix) and n != prefix + 'manifest.json' and not n.endswith('/')}
        contents[sidecar] = z.read(prefix + 'manifest.json')
    return contents


def build():
    contents = frozen(ROOT / f'notebooks/kaggle/stage2_repair_hints/{HINTS}.zip', HINTS + '/',
                      '_stage2_repair_hints_manifest.json')
    for p in source_files():
        rel = str(p.relative_to(ROOT))
        if rel in contents:
            if contents[rel] != p.read_bytes():
                raise ValueError(f'Frozen source differs from the completed repair-hints pack: {rel}')
        elif rel.startswith('eval/stage2_baselines_hints/'):
            contents[rel] = p.read_bytes()
        else:
            raise ValueError(f'Unexpected new runtime file outside eval/stage2_baselines_hints: {rel}')
    manifest = json.dumps(dict(experiment=EXPERIMENT, sha256={n: hashlib.sha256(b).hexdigest()
                                                              for n, b in sorted(contents.items())}),
                          sort_keys=True, indent=2).encode()
    sha = hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE / f'{NAME}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(f'{NAME}/manifest.json', manifest)
        for n, b in sorted(contents.items()):
            z.writestr(f'{NAME}/{n}', b)

    template = nb.read(ROOT / 'notebooks/kaggle/stage2_repair_hints/stage2_repair_hints_selective_kaggle.ipynb',
                       as_version=4)
    base_header = template.cells[1].source.split('PIPELINE=')[0]
    loader = 'MANIFEST_SHA=' + repr(sha) + '\n' + template.cells[2].source.split('\n', 1)[1]
    loader = (loader.replace(HINTS, NAME)
              .replace("=='stage2-repair-hints-3b-fp16-v1'", f"=='{EXPERIMENT}'")
              .replace("'_stage2_repair_hints_manifest.json'", f"'{MANIFEST}'")
              .replace("WORK/'stage2_repair_hints_packages'", "WORK/'stage2_baselines_hints_packages'"))
    assert 'repair_hints' not in loader and NAME in loader, 'Loader still refers to the old pack'
    prepare_all = template.cells[3].source
    assert 'path_file=' in prepare_all and '--method' in prepare_all
    prepare_no_adapter = prepare_all.split('path_file=')[0] + 'save_results()\n'
    for name, spec in RUNS.items():
        bundle = f'stage2_baselines_hints_{name}'
        title, what, compare = TITLES[name]
        if spec['adapter']:
            zip_name = ADAPTERS[spec['method']].split()[0]
            header = (base_header + f"RUN={name!r};METHOD={spec['method']!r}\n"
                      f"for name in ['{NAME}.zip','{zip_name}']:\n"
                      "    matches=list(INPUT.rglob(name));print(('found ZIP: ' if matches else 'No intact ZIP (may be extracted): ')+name,*matches)\n"
                      "print('The next cells verify source files and adapter contents, including automatically extracted datasets.')\n")
            prepare, adapter_arg = prepare_all, "'--adapter',str(ADAPTER),"
            inputs = f'**{NAME}.zip** and the same adapter as the proposed run: {ADAPTERS[spec["method"]]}'
        else:
            header = (base_header + f"RUN={name!r}\n"
                      f"matches=list(INPUT.rglob('{NAME}.zip'));print('found ZIP:' if matches else 'No intact ZIP (may be extracted):','{NAME}.zip',*matches)\n")
            prepare, adapter_arg = prepare_no_adapter, ''
            inputs = f'**{NAME}.zip** only (no parser adapter)'
        execution = f'''DEV_COUNT=20
def execute(smoke,split='test'):
    name='{bundle}'+('_smoke' if smoke else '_dev' if split=='dev' else '')
    out=RUN_DIR/name;assert not out.exists(),'Use a fresh run folder.'
    command=[sys.executable,'-m','eval.stage2_baselines_hints.runner','--run',RUN,{adapter_arg}'--out',str(out),'--packages',str(PACKAGES),'--split',split]
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
print('COMPLETE: download the full ZIP and this executed notebook for analysis (tools/analyze_baselines_hints_run.py).')
'''
        intro = f'''# Stage 2 — {title}
{what}

Everything else is the current 3B stack used for the repair-hints proposed runs, executed from the same frozen code: Qwen2.5-3B FP16 generator and repair model, prompts, budgets, sandbox and scoring. Reference programs and outputs remain evaluator-only. Compare with: {compare}.

Inputs: {inputs}. Enable Internet, select T4 x2, then Save Version → Run All (smoke, 20 dev, then 100 test tasks). Download **{bundle}.zip** and the executed notebook.

The contract check and repair hints were designed after inspecting failures on these test tasks, so test results are diagnostic, not independent evidence.
'''
        book = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(intro)] +
                                  [nb.v4.new_code_cell(c) for c in (header, loader, prepare, execution, finish)],
                                  metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                           'name': 'python3'}, 'language_info': {'name': 'python'},
                                            'experiment': EXPERIMENT})
        nb.validate(book)
        for c in book.cells:
            if c.cell_type == 'code':
                compile(c.source, f'{name}-notebook', 'exec')
        nb.write(book, HERE / f'stage2_baselines_hints_{name}_kaggle.ipynb')
    print(sha, len(contents))
    return sha


if __name__ == '__main__':
    build()
