"""Oracle boundary, frozen repair parity, packaging and comparison checks."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import nbformat
import pytest
from tests.test_stage2_combined_3b import factory
from eval.stage2_combined_gold.runner import run
from eval.stage2_combined_gold.collect import read_bundle,collect
from eval.stage2_combined_gold.artifacts import export
ROOT=Path(__file__).resolve().parents[1]


def test_gold_fresh_generation_guard_repair_no_parser(tmp_path):
    backend,live,calls,lifecycle=factory('selective')
    out=run('gold',tmp_path/'gold',smoke=True,backend_factory=backend)
    plan,rows=read_bundle(out.with_suffix('.zip'),allow_scripted=True)
    assert all(r['final_correct'] and not r['initial_correct'] and r['attempts']==1 for r in rows)
    assert all(r['parser_calls']==r['parser_tokens']==0 for r in rows)
    assert not live and lifecycle==[(op,role) for role in ['generator','agent'] for op in ['load','close']]
    assert plan['contract_source']=='task.gold_contract' and plan['adapter_hashes']=={}
    assert plan['common']['repair_arm']=='combined'
    assert all('reference_code' not in str(msg) for role,msg in calls)
    assert len([x for x in calls if x[0]=='generator'])==2
    assert len(list((out/'contracts').glob('*.json')))==2
    with pytest.raises(ValueError,match='real-model'):read_bundle(out.with_suffix('.zip'))
    with pytest.raises(ValueError,match='fresh output'):run('gold',out,smoke=True,backend_factory=backend)
    contract=next((out/'contracts').glob('*.json'));obj=json.loads(contract.read_text());obj['extracted']['contract']={};contract.write_text(json.dumps(obj));export(out,True)
    with pytest.raises(ValueError,match='Gold contract differs'):read_bundle(out.with_suffix('.zip'),allow_scripted=True)


def test_paired_gold_predicted_uses_same_settings(tmp_path):
    from eval.stage2_combined_3b.runner import run as predicted_run
    backend,*_=factory('selective');g=run('gold',tmp_path/'gold',smoke=True,backend_factory=backend)
    backend,*_=factory('selective');p=predicted_run('selective',tmp_path/'predicted',smoke=True,backend_factory=backend)
    result=collect(g.with_suffix('.zip'),tmp_path/'comparison',[p.with_suffix('.zip')],allow_scripted=True)
    assert result['summary']['gold']['correct']==2
    assert result['summary']['selective']['correct']==2
    assert result['paired']['selective']==dict(only_gold_correct=[],only_predicted_correct=[])
    assert not result['model_quality_result']


@pytest.mark.parametrize('packed',[True,False])
def test_notebook_loader_source_seal_and_python_cells(tmp_path,monkeypatch,packed):
    here=ROOT/'notebooks/kaggle/stage2_combined_gold';inputs=tmp_path/'input';inputs.mkdir();work=tmp_path/'work'
    source=here/'stage2_combined_gold_source.zip'
    if packed:shutil.copy(source,inputs/source.name)
    else:
        with zipfile.ZipFile(source) as z:z.extractall(inputs)
    book=nbformat.read(here/'stage2_combined_gold_3b_kaggle.ipynb',as_version=4)
    assert book.metadata.language_info.name=='python'
    for c in book.cells:
        if c.cell_type=='code':compile(c.source,'gold-notebook','exec')
    assert '--adapter' not in book.cells[4].source
    assert '--method\',\'gold' in book.cells[4].source
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:None)
    ns={};exec(book.cells[1].source.replace("INPUT=Path('/kaggle/input')",f'INPUT=Path({str(inputs)!r})').replace("WORK=Path('/kaggle/working')",f'WORK=Path({str(work)!r})'),ns)
    exec(book.cells[2].source,ns)
    root=ns['ROOT'];manifest=json.loads((root/'_stage2_combined_gold_manifest.json').read_text())
    assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in manifest['sha256'].items())
    with zipfile.ZipFile(ROOT/'notebooks/kaggle/stage2_combined_3b/stage2_combined_3b_source.zip') as z:
        assert (root/'_stage2_combined_3b_manifest.json').read_bytes()==z.read('stage2_combined_3b_source/manifest.json')
