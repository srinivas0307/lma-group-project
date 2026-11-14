"""Scope correction, unchanged prompts/weights and inference-only packaging."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile
import nbformat
import pandas as pd
import pytest
from finetune.selective_facts_scopefix.context import selective_facts,messages,FactUnavailable
from finetune.selective_facts_scopefix.preflight import check
from finetune.selective_facts_scopefix import adapter as adapter_module
from finetune.selective_facts.provenance import ROOT,digest

@pytest.mark.parametrize('kind',['median','mean','mode'])
def test_null_policy_when_is_not_a_row_scope(kind):
    df=pd.DataFrame({'score':[1.123456789,3.987654321,3.987654321,None]})
    request=f'Fill missing score with its full-column {kind}. Ignore missing values when computing statistics.'
    fact=selective_facts(request,df)[0]
    expected=df.score.mode().iloc[0] if kind=='mode' else getattr(df.score,kind)()
    assert fact['value']==expected and fact['scope']=={'kind':'full_column'}
    msg=messages(dict(request=request,data_file='input.csv'),df)
    assert msg[-1]['content'].endswith('Request: '+request)


def test_real_conditions_and_unavailable_facts_still_rejected():
    df=pd.DataFrame({'score':[1.,9.,None,1000.],'zone':['east','east','east','west']})
    policy=' Ignore missing values when computing statistics.'
    fact=selective_facts('Fill missing score only in zone = east with the median of score among rows with zone = east.'+policy,df)[0]
    assert fact['value']==5. and fact['scope']['kind']=='conditional'
    fact=selective_facts('Fill missing score only in zone = east with the full-column median of score.'+policy,df)[0]
    assert fact['value']==9. and fact['scope']['kind']=='full_column'
    for text in ['Fill missing score with its median when zone is east.','Fill missing score with its median only in zone = east.','Fill missing score with the median of positive values.']:
        with pytest.raises(FactUnavailable):selective_facts(text+policy,df)
    with pytest.raises(FactUnavailable,match='tied'):
        selective_facts('Fill missing score with its mode.'+policy,pd.DataFrame({'score':['a','b',None]}))
    with pytest.raises(FactUnavailable,match='No nonmissing'):
        selective_facts('Fill missing score with its median.'+policy,pd.DataFrame({'score':[None,None]}))


def test_all_twelve_repaired_and_eighty_eight_prompts_unchanged():
    result=check()
    assert len(result['repaired'])==12 and len(result['unchanged'])==88
    from finetune.selective_facts.evaluate import development_tasks
    for task,cohort,root in development_tasks():
        if task['id'] not in result['repaired']:continue
        fact=selective_facts(task['request'],pd.read_csv(root/task['data_file']))[0]
        literal=next(p['args'][0]['lit'] for p in task['gold_contract']['must_change'] if p.get('fn')=='const')
        if isinstance(literal,float):assert fact['value']==pytest.approx(literal,rel=0,abs=1e-12)
        else:assert fact['value']==literal


def fake_adapter(tmp_path,monkeypatch):
    contents={'adapter_config.json':b'{}','adapter_model.safetensors':b'weights','input_protocol.json':b'{}','selection.json':b'{}','igt_prompt_version.txt':b'p1-v2+sf1'}
    monkeypatch.setattr(adapter_module,'SPEC',dict(sha256={n:hashlib.sha256(b).hexdigest() for n,b in contents.items()}))
    monkeypatch.setattr(adapter_module,'validate_adapter',lambda folder:{n:digest(Path(folder)/n) for n in contents})
    return contents

@pytest.mark.parametrize('packed',[False,True])
def test_adapter_discovery_packed_or_extracted_never_changes_inputs(tmp_path,monkeypatch,packed):
    contents=fake_adapter(tmp_path,monkeypatch);inputs=tmp_path/'inputs';inputs.mkdir();work=tmp_path/'work'
    if packed:
        with zipfile.ZipFile(inputs/'selective_facts_1_5b.zip','w') as z:
            for n,b in contents.items():z.writestr('adapter/'+n,b)
    else:
        folder=inputs/'adapter';folder.mkdir()
        for n,b in contents.items():(folder/n).write_bytes(b)
    before={str(p):p.read_bytes() for p in inputs.rglob('*') if p.is_file()}
    found=adapter_module.discover(inputs,work)
    assert adapter_module.validate_existing(found)
    assert before=={str(p):p.read_bytes() for p in inputs.rglob('*') if p.is_file()}
    (found/'adapter_model.safetensors').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='epoch-2'):adapter_module.validate_existing(found)


def test_comparison_uses_matching_tasks_and_marks_smoke():
    from finetune.selective_facts_scopefix.comparison import compare
    old=json.loads((ROOT/'finetune/selective_facts_scopefix/prior_evaluation.json').read_text())['sf1']['records']
    tid=next(k for k,v in old.items() if v['parse_status']=='fact_unavailable')
    r=dict(old[tid],task_id=tid,parse_status='ok',all_correct_accepted=True,any_wrong_accepted=False,prompt_tokens=10,output_tokens=5,facts=[{'value':123}])
    result=compare([r],smoke=True)
    assert tid in result['paired']['sf1']['gained'] and result['routing']['previously_blocked'][tid]['facts_delivered']
    assert not result['passed_original_screen']
    with pytest.raises(ValueError,match='same 100'):compare([r])
    with pytest.raises(ValueError,match='identity'):compare([dict(r,request='different')],smoke=True)


def test_inference_evaluator_with_scripted_answers(tmp_path,monkeypatch):
    from finetune.selective_facts_scopefix.evaluate import evaluate
    from finetune.selective_facts.evaluate import development_tasks
    from igt.agents.intent_parser import contract_to_text
    import finetune.selective_facts.backend as backend
    import finetune.selective_facts.attention as attention
    answers={task['request']:task for task,_,_ in development_tasks(True)}
    seen=[]
    class Fake:
        def __init__(self,*args):pass
        def complete(self,msg,*args):
            task=next(t for req,t in answers.items() if msg[-1]['content'].endswith('Request: '+req))
            if task['pattern']=='fill_0':assert 'Required input facts' in msg[-1]['content']
            seen.append(task['id']);return contract_to_text(task['gold_contract']),100,50
        def close(self):pass
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace())
    monkeypatch.setattr(backend,'Backend',Fake)
    monkeypatch.setattr(attention,'configure_attention',lambda *args:None)
    monkeypatch.setattr(adapter_module,'validate_existing',lambda folder:{'verified':'yes'})
    folder=tmp_path/'adapter';folder.mkdir();out=tmp_path/'result'
    result=evaluate(folder,out,smoke=True)
    assert len(seen)==2 and result['summary']['synthetic']['correct_accepted']==2
    assert result['provenance']['adapter_trained_protocol']=='p1-v2+sf1'
    assert result['provenance']['protocol']=='p1-v2+sf1-scopefix1'
    assert json.loads((out/'comparison.json').read_text())['routing']['fact_unavailable']==0

@pytest.mark.parametrize('packed',[False,True])
def test_notebook_source_loader_in_both_kaggle_layouts(tmp_path,monkeypatch,packed):
    here=ROOT/'notebooks/kaggle/selective_facts_scopefix'
    b=nbformat.read(here/'selective_facts_scopefix_eval.ipynb',as_version=4);nbformat.validate(b)
    inputs=tmp_path/'inputs';inputs.mkdir();work=tmp_path/'work'
    if packed:(inputs/'selective_facts_scopefix_source.zip').write_bytes((here/'selective_facts_scopefix_source.zip').read_bytes())
    else:
        with zipfile.ZipFile(here/'selective_facts_scopefix_source.zip') as z:z.extractall(inputs)
    monkeypatch.setattr(subprocess,'run',lambda *args,**kwargs:None)
    setup=b.cells[1].source.replace('/kaggle/input',str(inputs)).replace('/kaggle/working',str(work));namespace={}
    exec(compile(setup,'setup','exec'),namespace)
    source=namespace['ROOT']
    assert (source/'finetune/selective_facts_scopefix/context.py').exists()
    assert not (source/'benchmark/contract_quality_v1/selection.jsonl').exists()
    assert not (source/'finetune/data/selective_facts_v1').exists()
    for cell in b.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook','exec')
    assert b.metadata.language_info.name=='python'
