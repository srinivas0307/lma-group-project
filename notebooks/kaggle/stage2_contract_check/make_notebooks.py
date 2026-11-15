"""Build the sealed contract-check-v1 source pack and one Kaggle notebook per parser method.

The completed combined 3B pack is reused byte for byte; only the new eval/stage2_contract_check modules
(plus any runtime file they import that the old pack lacks) are added.
    .venv/bin/python -m notebooks.kaggle.stage2_contract_check.make_notebooks
"""
import hashlib
import json
import zipfile
from pathlib import Path

import nbformat as nb

from eval.stage2_contract_check.artifacts import ROOT, source_files

HERE = Path(__file__).resolve().parent
NAME = 'stage2_contract_check_source'
EXPERIMENT = 'stage2-contract-check-3b-fp16-v1'


def build():
    original = ROOT / 'notebooks/kaggle/stage2_combined_3b'
    with zipfile.ZipFile(original / 'stage2_combined_3b_source.zip') as z:
        prefix = 'stage2_combined_3b_source/'
        contents = {n[len(prefix):]: z.read(n) for n in z.namelist()
                    if n.startswith(prefix) and n != prefix + 'manifest.json' and not n.endswith('/')}
        contents['_stage2_combined_3b_manifest.json'] = z.read(prefix + 'manifest.json')
    for p in source_files():
        rel = str(p.relative_to(ROOT))
        if rel in contents:
            if contents[rel] != p.read_bytes():
                raise ValueError(f'Frozen source differs from the completed pack: {rel}')
        else:
            contents[rel] = p.read_bytes()
    manifest = json.dumps(dict(experiment=EXPERIMENT, sha256={n: hashlib.sha256(b).hexdigest()
                                                              for n, b in sorted(contents.items())}),
                          sort_keys=True, indent=2).encode()
    sha = hashlib.sha256(manifest).hexdigest()
    with zipfile.ZipFile(HERE / f'{NAME}.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(f'{NAME}/manifest.json', manifest)
        for n, b in sorted(contents.items()):
            z.writestr(f'{NAME}/{n}', b)
    for method in ('selective', 'tools'):
        old = nb.read(original / f'stage2_combined_3b_{method}_proposed_kaggle.ipynb', as_version=4)
        header = old.cells[1].source.replace("'stage2_combined_3b_source.zip'", f"'{NAME}.zip'")
        loader = 'MANIFEST_SHA=' + repr(sha) + '\n' + old.cells[2].source.split('\n', 1)[1]
        loader = (loader.replace("ROOT=WORK/('stage2_combined_3b_source_'", f"ROOT=WORK/('{NAME}_'")
                  .replace("=='stage2-combined-3b-fp16-v1'", f"=='{EXPERIMENT}'")
                  .replace("'_stage2_combined_3b_manifest.json'", "'_stage2_contract_check_manifest.json'")
                  .replace("INPUT.rglob('stage2_combined_3b_source.zip')", f"INPUT.rglob('{NAME}.zip')")
                  .replace("prefix='stage2_combined_3b_source/'", f"prefix='{NAME}/'")
                  .replace("'Attach/place stage2_combined_3b_source.zip", f"'Attach/place {NAME}.zip")
                  .replace("WORK/'stage2_combined_3b_packages'", "WORK/'stage2_contract_check_packages'"))
        assert 'stage2_combined_3b_source' not in loader, 'Loader still refers to the old pack'
        prepare = old.cells[3].source
        execution = (old.cells[4].source
                     .replace("'stage2_combined_3b_'+METHOD+'_proposed'", "'stage2_contract_check_'+METHOD+'_checked'")
                     .replace("'eval.stage2_combined_3b.runner','--method',METHOD,'--preset','proposed',",
                              "'eval.stage2_contract_check.runner','--method',METHOD,'--preset','proposed','--arm','checked',"))
        assert "'--arm','checked'" in execution
        finish = old.cells[5].source.replace(
            "Run the other method independently, then the CPU collector.",
            "Send it back for paired analysis against the completed combined 3B control (tools/analyze_contract_check_run.py).")
        intro = f'''# Stage 2 — {method} parser + contract checks (contract-check-v1) + combined 3B repair
Same pipeline, models, adapters, prompts and limits as the completed combined 3B {method} run (74 correct / 1 incorrect accepted / 25 withheld). One change: after the unchanged first parse, input-only checks inspect the contract. When a check fires, the same parser gets **one** correction turn with the issues. A contract that still has a proven inconsistency is withheld with its reason; evidence-only issues never cause withholding. Checks use only the request, input data and the predicted contract; reference programs and gold contracts remain evaluator-only.

Inputs: **{NAME}.zip** and the same selected adapter dataset as the completed run ({'selective_facts_1_5b.zip' if method == 'selective' else 'tool_use_train_1_5b.zip'}). Enable Internet, select T4 x2, then Save Version → Run All. The notebook runs a two-task smoke, 20 development tasks, then 100 diagnostic test tasks, each fresh. Download **stage2_contract_check_{method}_checked.zip** and the executed notebook.

The checks were designed after inspecting failures on these test tasks, so test gains are diagnostic, not independent evidence. Records keep the first and final contracts, every issue, the correction prompt/reply, its tokens and the decision.
'''
        book = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(intro)] +
                                  [nb.v4.new_code_cell(c) for c in (header, loader, prepare, execution, finish)],
                                  metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                           'name': 'python3'}, 'language_info': {'name': 'python'},
                                            'experiment': EXPERIMENT})
        nb.validate(book)
        for c in book.cells:
            if c.cell_type == 'code':
                compile(c.source, f'{method}-notebook', 'exec')
        nb.write(book, HERE / f'stage2_contract_check_{method}_kaggle.ipynb')
    print(sha, len(contents))
    return sha


if __name__ == '__main__':
    build()
