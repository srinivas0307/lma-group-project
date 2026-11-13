import json
import sys
from pathlib import Path
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from eval.contract_quality.build_suite import build,SUITE
from eval.contract_quality.context import profile,messages
from eval.contract_quality.harness import load_tasks,score

def test_splits_and_determinism(tmp_path):
    build(tmp_path)
    dev=load_tasks(tmp_path);selection=load_tasks(tmp_path,'selection',True)
    assert len(dev)==80 and len(selection)==40
    assert not {t['dataset'] for t in dev}&{t['dataset'] for t in selection}
    assert {t['family'] for t in dev}=={t['family'] for t in selection}
    assert (tmp_path/'dev.jsonl').read_bytes()==(SUITE/'dev.jsonl').read_bytes()
    with pytest.raises(ValueError,match='locked'):load_tasks(tmp_path,'selection')
    (tmp_path/'data/warehouse.csv').write_text('tampered')
    with pytest.raises(ValueError,match='hash mismatch'):load_tasks(tmp_path)

def test_profile_full_precision_null_ties_and_group_scope():
    df=pd.DataFrame({'value':[1.123456789,None,3.987654321,10.0], 'group':['a','a','a','b']})
    p=json.loads(profile(df))
    assert p['columns']['value']['median']==3.987654321
    assert p['groups']['group'][0]['numeric']['value']['median']==pytest.approx(2.555555555)
    ties=json.loads(profile(pd.DataFrame({'label':['a','b',None]})))
    assert ties['columns']['label']['modes']==['a','b']
    with pytest.raises(ValueError,match='budget exceeded'):profile(df,max_chars=10)

def test_messages_ignore_gold_and_reference():
    task=load_tasks()[0];df=pd.read_csv(SUITE/task['data_file'])
    original=messages(task,df,'original');grounded=messages(task,df,'grounded')
    changed=dict(task,gold_contract={'SECRET':'DO NOT INCLUDE'},reference_code='SECRET')
    assert messages(changed,df,'grounded')==grounded
    assert grounded[0]==original[0]
    assert grounded[1]['content'].startswith(original[1]['content'])

def test_invalid_not_success_and_duplicate_rejected():
    row=dict(task_id=load_tasks()[0]['id'],variant='original',raw_output='no contract')
    result=score([row])['summary']['original']
    assert result['invalid']==1 and result['joint_success']==0 and not result['complete']
    with pytest.raises(ValueError,match='Duplicate'):score([row,row])

def test_portable_notebooks_and_embedded_sources(tmp_path):
    import ast,base64,hashlib,zlib,nbformat
    import subprocess
    for size in ['1_5b','3b']:
        path=ROOT/f'notebooks/kaggle/contract_quality/cq_{size}_pilot.ipynb'
        book=nbformat.read(path,as_version=4);nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type=='code':compile(cell.source,str(path),'exec')
        constants={}
        for node in ast.parse(book.cells[1].source).body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['PAYLOAD','PAYLOAD_SHA']:
                constants[node.targets[0].id]=ast.literal_eval(node.value)
        raw=zlib.decompress(base64.b64decode(constants['PAYLOAD']))
        assert hashlib.sha256(raw).hexdigest()==constants['PAYLOAD_SHA']
        files=json.loads(raw)
        for name,source in files.items():
            assert (ROOT/name).read_text()==source
            dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(source)
        result=subprocess.run([sys.executable,'-m','eval.contract_quality.pilot','--help'],cwd=tmp_path,capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_collector_rejects_smoke_and_mixed_inventory(tmp_path):
    import hashlib,zipfile
    from eval.contract_quality.collect import read_bundle
    spec=json.loads((SUITE/'adapters.json').read_text())['1_5b']
    plan=dict(smoke=False,experiment='cq-draft-v1',split='dev',variants=['original','grounded'],size='1_5b',
              base=spec['model'],adapter_hashes=spec['sha256'],suite_manifest=hashlib.sha256((SUITE/'manifest.json').read_bytes()).hexdigest())
    rows=[dict(task_id=t['id'],variant=v,raw_output='invalid') for t in load_tasks() for v in ['original','grounded']]
    def write(smoke=False):
        path=tmp_path/'bundle.zip'
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('plan.json',json.dumps(plan))
            z.writestr('completion.json',json.dumps(dict(smoke=smoke,complete=True,records=160)))
            z.writestr('predictions.jsonl',''.join(json.dumps(r)+'\n' for r in rows))
        return path
    assert len(read_bundle(write())[1])==160
    with pytest.raises(ValueError,match='non-smoke'):read_bundle(write(True))
    rows[0]['task_id']='old_test_id'
    with pytest.raises(ValueError,match='coverage'):read_bundle(write())

def test_semantic_probe_detects_vacuous_contract():
    from eval.contract_quality.harness import prepare,probe
    t=load_tasks()[0]
    prepared=prepare(t)
    gold=probe(t['gold_contract'],prepared)
    assert gold['all_correct_accepted'] and not gold['any_wrong_accepted']
    empty=probe({'contract_version':'1.0','must_change':[],'must_preserve':[]},prepared)
    assert empty['all_correct_accepted'] and empty['any_wrong_accepted']
