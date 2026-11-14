import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile
import nbformat
import pandas as pd
import pytest
from finetune.selective_facts_correction.checks import normalize,diagnose
from finetune.selective_facts_correction.parser import attempt_correction
from igt.agents.intent_parser import extract_contract

ROOT=Path(__file__).resolve().parents[1]
DF=pd.DataFrame({'x':[1.,2.,None],'label':['low','mid',None]})
TASK=dict(request='Add 2 to x and preserve other columns.',data_file='input.csv')
VALID=dict(contract_version='1.0',must_change=[dict(type='column_value_relation',target='x',fn='add',args=[{'col':'x'},{'lit':2}])],must_preserve=[dict(type='row_count',op='unchanged'),dict(type='columns_unchanged',target='*',**{'except':['x']})])

class Fake:
    def __init__(self,replies):self.replies=list(replies);self.calls=[]
    def complete(self,messages,*args):
        self.calls.append(copy.deepcopy(messages))
        if not self.replies:raise AssertionError('Extra model call')
        return self.replies.pop(0),20,10

def test_normalize_only_operator_positions_preserves_everything_else():
    c=copy.deepcopy(VALID)
    c['extra']='do not silently delete this'
    c['must_change'][0]['args'][1]={'lit':{'op':' > ','column':' x '}}
    c['must_change'][0]['where']={'not':{'column':'x','op':' > ','value':1}}
    c['must_preserve'][0]['op']=' unchanged '
    n=normalize(json.dumps(c))
    assert len(n['changes'])==2
    expected=copy.deepcopy(c);expected['must_change'][0]['where']['not']['op']='>';expected['must_preserve'][0]['op']='unchanged'
    assert n['extracted']['contract']==expected
    assert n['extracted']['status']=='invalid_contract' # extra fields remain invalid
    assert n['extracted']['contract']['must_change'][0]['args'][1]['lit']['op']==' > '

@pytest.mark.parametrize('text',['not JSON','{"contract_version":"1.0",}','{"must_change":[],"must_preserve":[]}'])
def test_no_changes_preserve_exact_raw_text(text):
    n=normalize(text);assert n['text']==text and not n['changes']

def test_unknown_operator_not_converted_and_wrong_version_not_fixed():
    c=copy.deepcopy(VALID);c['contract_version']='2.0';c['must_preserve'][0]['op']=' eq ';c['must_preserve'][0]['value']=3
    n=normalize(json.dumps(c));assert n['extracted']['status']=='invalid_contract';assert n['extracted']['contract']['contract_version']=='2.0'
    c['must_preserve'][0]['op']=' <> ';assert not normalize(json.dumps(c))['changes']

def test_unknown_names_and_existing_addition_are_feedback_only():
    c=copy.deepcopy(VALID);c['must_change'][0]['args'][0]['col']='xx'
    original=copy.deepcopy(c);issues=diagnose(extract_contract(json.dumps(c)),DF)
    assert any(i['code']=='unknown_column' for i in issues);assert c==original
    c=copy.deepcopy(VALID);c['must_change'].append({'type':'columns_added','target':['x']})
    assert any(i['code']=='existing_column_added' for i in diagnose(extract_contract(json.dumps(c)),DF))

def test_contradiction_requires_changed_overlapping_retained_cells():
    c=copy.deepcopy(VALID);c['must_preserve'].append({'type':'column_values_unchanged','target':'x'})
    assert any(i['code']=='change_preserve_conflict' for i in diagnose(extract_contract(json.dumps(c)),DF))
    c['must_preserve'][-1]['where']={'column':'x','op':'is_null'}
    assert not diagnose(extract_contract(json.dumps(c)),DF)
    c['must_preserve'][-1].pop('where');c['must_preserve']=c['must_preserve'][1:] # no promised retained rows
    assert not diagnose(extract_contract(json.dumps(c)),DF)

def test_removed_rows_not_used_to_infer_contradiction():
    c=copy.deepcopy(VALID)
    c['must_change'] += [{'type':'rows_removed_where','condition':{'column':'x','op':'not_null'}}]
    c['must_preserve'] += [{'type':'column_values_unchanged','target':'x'},{'type':'rows_retained_where','condition':{'column':'x','op':'is_null'}}]
    assert not diagnose(extract_contract(json.dumps(c)),DF)

def test_one_call_when_valid_and_no_gold_leak():
    b=Fake([json.dumps(VALID)])
    result=attempt_correction(dict(TASK,gold_contract='SECRET',reference_code='SECRET',task_id='SECRET'),DF,b)
    assert len(b.calls)==1 and not result['trace']['attempted']
    assert 'SECRET' not in str(b.calls)
    assert result['variants']['corrected']['status']=='ok'

def test_normalized_valid_contract_avoids_retry():
    c=copy.deepcopy(VALID);c['must_preserve'][0]['op']=' unchanged '
    b=Fake([json.dumps(c)]);r=attempt_correction(TASK,DF,b)
    assert len(b.calls)==1 and r['variants']['control']['status']=='invalid_contract' and r['variants']['normalized']['status']=='ok'

def test_one_retry_success_rejected_failure_no_fallback():
    c=copy.deepcopy(VALID);c['must_change'].append({'type':'columns_added','target':['x']})
    for second,expected in [(VALID,'ok'),(c,'correction_failed')]:
        b=Fake([json.dumps(c),json.dumps(second)])
        r=attempt_correction(TASK,DF,b)
        assert len(b.calls)==2 and r['trace']['attempted']
        assert 'existing input columns' in b.calls[1][-1]['content']
        assert r['variants']['corrected']['status']==expected
        if expected!='ok':assert r['variants']['corrected']['contract'] is None

def test_context_guard_and_gpu_errors_not_hidden():
    b=Fake(['bad'])
    def no_room(*a):raise ValueError('context too large')
    b.check_prompt=no_room
    r=attempt_correction(TASK,DF,b);assert len(b.calls)==1 and r['variants']['corrected']['status']=='correction_context_limit'
    b=Fake(['bad'])
    def failure(*a):raise RuntimeError('CUDA OOM')
    b.check_prompt=failure
    with pytest.raises(RuntimeError,match='CUDA'):attempt_correction(TASK,DF,b)

def test_fact_failure_does_not_retry_or_generate():
    b=Fake([]);r=attempt_correction(dict(TASK,request='Fill missing x with its median among positive values.'),DF,b)
    assert not b.calls and r['variants']['corrected']['status']=='fact_unavailable'

def test_gold_contracts_no_flags_and_saved_passing_cases_untouched():
    from finetune.selective_facts.evaluate import development_tasks
    path=ROOT/'kaggle_outputs/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval.zip'
    with zipfile.ZipFile(path) as z:rows={r['task_id']:r for n in z.namelist() if n.startswith('tasks/') and (r:=json.loads(z.read(n)))}
    normalized=[];flagged=[];correct=0
    for task,cohort,root in development_tasks():
        before=pd.read_csv(root/task['data_file']);r=rows[task['id']];n=normalize(r['raw_output']);issues=diagnose(n['extracted'],before)
        if n['changes']:normalized.append(task['id'])
        if issues:flagged.append(task['id'])
        if r['all_correct_accepted']:
            correct+=1;assert not n['changes'] and not issues
        if cohort=='synthetic':assert not diagnose(extract_contract(json.dumps(task['gold_contract'])),before)
    assert correct==85 and len(normalized)==3 and len(flagged)==10


def test_paired_evaluation_through_real_verifier(tmp_path,monkeypatch):
    from finetune.selective_facts_correction import evaluate as ev
    import finetune.selective_facts.backend as backend
    import finetune.selective_facts.attention as attention
    tasks=ev.task_set(True)
    gold={t['request']:t['gold_contract'] for t,_,_ in tasks}
    class Backend(Fake):
        calls_all=[]
        def __init__(self,*args):pass
        def complete(self,msg,*args):
            req=next(k for k in gold if msg[1]['content'].endswith('Request: '+k));c=copy.deepcopy(gold[req])
            if len(msg)==2:
                if req.startswith('Keep only'):
                    def bad(node):
                        if isinstance(node,dict):
                            for k,v in node.items():
                                if k=='op' and v=='>':node[k]=' > '
                                else:bad(v)
                        elif isinstance(node,list):
                            for x in node:bad(x)
                    bad(c)
                elif req.startswith('Map '):c['must_preserve'].append({'type':'columns_unchanged','target':'*'})
            self.calls_all.append(msg);return json.dumps(c),100,50
        def close(self):pass
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(version=types.SimpleNamespace(cuda='mock'),cuda=types.SimpleNamespace(get_device_name=lambda i:'mock',device_count=lambda:1)))
    monkeypatch.setattr(backend,'Backend',Backend);monkeypatch.setattr(attention,'configure_attention',lambda *a:None)
    monkeypatch.setattr(ev,'validate_existing',lambda folder:{'mock':'hash'})
    monkeypatch.setattr(ev.importlib.metadata,'version',lambda n:'mock')
    result=ev.run(tmp_path/'adapter',tmp_path/'out',smoke=True)
    assert result['current']['control']['synthetic']['correct_accepted']==1
    assert result['current']['normalized']['synthetic']['correct_accepted']==2
    assert result['current']['corrected']['synthetic']['correct_accepted']==3
    assert len(Backend.calls_all)==4
    assert json.loads((tmp_path/'out/completion.json').read_text())['retrained'] is False
    assert (tmp_path/'out.zip').exists()
    with pytest.raises(ValueError,match='Existing'):ev.run(tmp_path/'adapter',tmp_path/'out',smoke=True)

@pytest.mark.parametrize('platform',['colab','kaggle'])
@pytest.mark.parametrize('packed',[True,False])
def test_notebooks_source_loader(tmp_path,monkeypatch,platform,packed):
    book=nbformat.read(ROOT/f'notebooks/{platform}/selective_facts_correction/selective_facts_correction_{platform}.ipynb',as_version=4)
    nbformat.validate(book)
    for c in book.cells:
        if c.cell_type=='code':compile(c.source,'notebook','exec')
    inputs=tmp_path/'input';inputs.mkdir();work=tmp_path/'work';work.mkdir()
    bundle=ROOT/'notebooks/kaggle/selective_facts_correction/selective_facts_correction_source.zip'
    if packed:(inputs/bundle.name).write_bytes(bundle.read_bytes())
    else:
        with zipfile.ZipFile(bundle) as z:z.extractall(inputs)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:None)
    namespace=dict(INPUT=inputs,WORK=work)
    exec('import hashlib,json,os,subprocess,sys,zipfile\nfrom pathlib import Path',namespace)
    exec(book.cells[2].source,namespace)
    assert (namespace['ROOT']/'finetune/selective_facts_correction/parser.py').exists()
    assert not (namespace['ROOT']/'finetune/data').exists()
    assert book.metadata.language_info.name=='python'
