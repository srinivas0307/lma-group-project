"""Fresh normalized parsers, sequential model lifecycle, genuine repair and strict comparison."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import nbformat
import pandas as pd
import pytest
from igt.agents.intent_parser import contract_to_text
from igt.agents.code_generator_v2 import build_generator_messages
from eval.stage2_dual_3b.parsers import parse
from eval.stage2_dual_3b.runner import run,load_tasks,SMOKE_IDS,CurrentRunGeneration
from eval.stage2_dual_3b.collect import collect,read_bundle
from eval.stage2_dual_3b.artifacts import export
from finetune.selective_facts_scopefix.context import schema_entry
ROOT=Path(__file__).resolve().parents[1]

class Fake:
    model='scripted'
    def __init__(self,replies):self.replies=list(replies);self.calls=[]
    def prompt_tokens(self,messages):return 0
    def complete(self,messages,*args,**kwargs):
        self.calls.append(copy.deepcopy(messages))
        if not self.replies:raise AssertionError('Unexpected model call')
        item=self.replies.pop(0)
        return item(messages) if callable(item) else item,100,50


def test_selective_normalization_has_one_generation_and_no_gold():
    df=pd.DataFrame({'x':[1.,3.,None]});task=dict(id='test',request='Fill missing x with its full-column median. Ignore missing values when computing statistics.',data_file='input.csv',gold_contract='SECRET',reference_code='SECRET')
    c=dict(contract_version='1.0',must_change=[dict(type='column_value_relation',target='x',fn='const',args=[{'lit':2.}],where=dict(column='x',op=' is_null '))],must_preserve=[])
    b=Fake([contract_to_text(c)]);r=parse('selective',task,df,b)
    assert r['extracted']['status']=='ok' and r['llm_calls']==1 and r['facts'][0]['value']==2
    assert r['normalizations'] and 'SECRET' not in str(b.calls)
    assert ' is_null ' in r['raw_output'] and r['extracted']['contract']['must_change'][0]['where']['op']=='is_null'


def test_tools_preserve_failed_calls_and_only_normalize_contracts():
    df=pd.DataFrame({'x':[1.,3.,None]});task=dict(id='test',request='Fill missing x with its median.',data_file='input.csv',reference_code='SECRET')
    c=dict(contract_version='1.0',must_change=[],must_preserve=[dict(type='row_count',op=' unchanged ')])
    def final(msg):
        obs=json.loads(msg[-1]['content'].splitlines()[1]);assert obs['result']['statistics']['median']==2
        return contract_to_text(c)
    b=Fake(['{"tool":"profile_column","args":{"column":"__unknown_column__"}}','{"tool":"profile_column","args":{"column":"x"}}',final])
    r=parse('tools',task,df,b)
    assert r['extracted']['status']=='ok' and r['tool_attempts']==2 and r['tool_failures']==1 and r['llm_calls']==3
    assert r['tool_calls'][0]['args']['column']=='__unknown_column__'
    assert r['normalizations'] and 'SECRET' not in str(b.calls)


def test_tool_and_selective_limits_and_errors():
    task=dict(id='x',request='Fill missing x with its median among positive values.',data_file='x.csv')
    df=pd.DataFrame({'x':[None,None]});b=Fake([])
    assert parse('selective',task,df,b)['extracted']['status']=='input_facts_unavailable' and not b.calls
    task['request']='Rename x to y.'
    b=Fake([])
    def fail(*a,**k):raise RuntimeError('CUDA OOM')
    b.complete=fail
    with pytest.raises(RuntimeError,match='CUDA'):parse('tools',task,df,b)


def pipeline_factory(method,wrong_initial=False):
    tasks={t['id']:t for _,t in load_tasks() if t['id'] in SMOKE_IDS}
    lifecycle=[];live=[];calls=[];tool_stages={}
    class Backend:
        def __init__(self,role):
            assert not live,'Simultaneous model residency';self.role=role;self.model='scripted-'+role;live.append(self);lifecycle.append(('load',role))
        def prompt_tokens(self,msg):return 0
        def complete(self,msg,*args,**kw):
            calls.append((self.role,copy.deepcopy(msg)))
            task=next(t for t in tasks.values() if ' '.join(t['request'].split()) in ' '.join(msg[1]['content'].split()))
            if self.role=='generator':
                code='def transform(df):\n    return df.copy()\n' if wrong_initial else task['reference_code']
                return '```python\n'+code+'\n```',100,50
            if self.role=='parser' or msg[0]['content'].startswith('You turn a data-transformation request'):
                if method=='tools' and self.role=='parser' and task['id']=='titanic_miss_004':
                    stage=tool_stages.get(task['id'],0);tool_stages[task['id']]=stage+1
                    if stage==0:return '{"tool":"profile_column","args":{"column":"__unknown_column__"}}',100,50
                    if stage==1:return '{"tool":"profile_column","args":{"column":"Age"}}',100,50
                c=copy.deepcopy(task['gold_contract'])
                for p in c['must_preserve']:
                    if p.get('type')=='row_count':p['op']=' '+p['op']+' '
                return contract_to_text(c),100,50
            return json.dumps(dict(action='propose_fix',code=task['reference_code'])),100,50
        def close(self):
            assert live==[self];live.clear();lifecycle.append(('close',self.role))
    return Backend,lifecycle,calls,live

@pytest.mark.parametrize('method',['selective','tools'])
def test_fresh_pipeline_one_resident_model_and_collector(tmp_path,method):
    factory,life,calls,live=pipeline_factory(method)
    out=run(method,tmp_path/method,smoke=True,backend_factory=factory)
    plan,rows=read_bundle(out.with_suffix('.zip'),allow_scripted=True)
    assert len(rows)==2 and all(r['final_correct'] for r in rows)
    assert life==[('load','parser'),('close','parser'),('load','generator'),('close','generator'),('load','agent'),('close','agent')]
    assert not live and len([x for x in calls if x[0]=='generator'])==2
    assert all(r['tokens']==r['parser_tokens']+r['generation_repair_tokens'] for r in rows)
    if method=='tools':assert sum(r['parser_tool_failures'] for r in rows)==1
    assert not list(out.glob('**/*prior*'))
    with pytest.raises(ValueError,match='real-model'):read_bundle(out.with_suffix('.zip'))
    with pytest.raises(ValueError,match='fresh output'):run(method,out,smoke=True,backend_factory=factory)
    for role,msg in calls:
        assert 'reference_code' not in str(msg) and 'gold_contract' not in str(msg)


def test_repair_phase_uses_actual_agent_and_initial_generation_once(tmp_path):
    factory,life,calls,live=pipeline_factory('selective',wrong_initial=True)
    out=run('selective',tmp_path/'repair',smoke=True,backend_factory=factory)
    _,rows=read_bundle(out.with_suffix('.zip'),allow_scripted=True)
    assert all(r['initial_correct'] is False and r['final_correct'] for r in rows)
    assert sum(role=='generator' for role,msg in calls)==2
    assert sum(role=='agent' for role,msg in calls)==2
    assert all(r['attempts']==1 for r in rows)


def test_invalid_parse_skips_generation_and_repair(tmp_path):
    b=Fake(['bad','bad'])
    def factory(role):
        assert role=='parser';return b
    out=run('selective',tmp_path/'invalid',smoke=True,backend_factory=factory)
    _,rows=read_bundle(out.with_suffix('.zip'),allow_scripted=True)
    assert all(r['stage_ended']=='parse' and r['outcome']=='ESCALATE' and r['generation_repair_tokens']==0 for r in rows)
    assert not list((out/'generation').glob('*'))


def test_comparison_and_corrupt_or_mixed_inputs_rejected(tmp_path):
    paths=[]
    for method in ['selective','tools']:
        factory,*_=pipeline_factory(method);out=run(method,tmp_path/method,smoke=True,backend_factory=factory);paths.append(out.with_suffix('.zip'))
    report=collect(paths,tmp_path/'summary',allow_scripted=True)
    assert report['summary']['selective']['correct']==2 and report['summary']['tools']['correct']==2
    assert not report['paired']['only_selective_correct']
    with pytest.raises(ValueError,match='each method'):collect([paths[0],paths[0]],tmp_path/'wrong',allow_scripted=True)
    with pytest.raises(ValueError,match='real-model'):collect(paths,tmp_path/'not-real')
    out=tmp_path/'tools';plan=json.loads((out/'plan.json').read_text());plan['common']['max_attempts']=99;(out/'plan.json').write_text(json.dumps(plan));export(out,True)
    with pytest.raises(ValueError,match='settings'):read_bundle(paths[1],allow_scripted=True)


def test_generation_buffer_is_exact_and_single_use():
    item=dict(model='x',messages=[{'role':'user','content':'q'}],raw_output='answer',prompt_tokens=3,output_tokens=2)
    replay=CurrentRunGeneration(item)
    with pytest.raises(ValueError,match='fresh matching'):replay.complete([{'role':'user','content':'other'}])
    assert replay.complete(item['messages'])==('answer',3,2)
    with pytest.raises(ValueError):replay.complete(item['messages'])

@pytest.mark.parametrize('platform',['colab','kaggle'])
@pytest.mark.parametrize('packed',[True,False])
def test_source_loader_and_all_notebook_cells(tmp_path,monkeypatch,platform,packed):
    folder=ROOT/f'notebooks/{platform}/stage2_dual_3b'
    for path in folder.glob('*.ipynb'):
        b=nbformat.read(path,as_version=4);nbformat.validate(b)
        for c in b.cells:
            if c.cell_type=='code':compile(c.source,str(path),'exec')
        assert b.metadata.language_info.name=='python'
    b=nbformat.read(folder/f'stage2_dual_3b_selective_proposed_{platform}.ipynb',as_version=4)
    inputs=tmp_path/'input';inputs.mkdir();work=tmp_path/'work';work.mkdir();bundle=ROOT/'notebooks/kaggle/stage2_dual_3b/stage2_dual_3b_source.zip'
    if packed:(inputs/bundle.name).write_bytes(bundle.read_bytes())
    else:
        with zipfile.ZipFile(bundle) as z:z.extractall(inputs)
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:None)
    ns=dict(INPUT=inputs,WORK=work)
    exec('import hashlib,json,os,subprocess,sys,zipfile\nfrom pathlib import Path',ns);exec(b.cells[2].source,ns)
    assert (ns['ROOT']/'eval/stage2_dual_3b/runner.py').exists()
    assert not (ns['ROOT']/'finetune/selective_facts_correction/historical.json').exists()
    assert not (ns['ROOT']/'finetune/selective_facts_scopefix/prior_evaluation.json').exists()
    assert not (ns['ROOT']/'results').exists() and not (ns['ROOT']/'finetune/data').exists()

@pytest.mark.parametrize('role,exception',[('parser','parser'),('agent','budget'),('generator','value')])
def test_declared_context_limit_refuses_explicitly(role,exception):
    from eval.stage2_dual_3b.backend import Backend,ContextLimitError
    from igt.llm import BudgetExceeded
    b=Backend.__new__(Backend);b.role=role;b.limit=5;b.refusals=[]
    class Tok:
        def apply_chat_template(self,*a,**kw):return 'prompt'
        def __call__(self,*a,**kw):return {'input_ids':[1]*5}
    b.tok=Tok()
    cls={'parser':ContextLimitError,'budget':BudgetExceeded,'value':ValueError}[exception]
    with pytest.raises(cls):b.check_prompt([],1)
    assert b.refusals[0]['reason']=='context_limit'


def test_missing_generation_record_cannot_be_collected(tmp_path):
    factory,*_=pipeline_factory('selective');out=run('selective',tmp_path/'source',smoke=True,backend_factory=factory)
    next((out/'generation').glob('*.json')).unlink();export(out,True)
    with pytest.raises(ValueError,match='initial generations'):read_bundle(out.with_suffix('.zip'),allow_scripted=True)


def test_retained_six_system_routes_and_oracle_boundary(tmp_path):
    from eval.stage2_dual_3b.config import SYSTEMS
    paths=[]
    for preset in SYSTEMS:
        factory,life,calls,live=pipeline_factory('selective')
        out=run('selective',tmp_path/preset,preset=preset,smoke=True,backend_factory=factory)
        plan,rows=read_bundle(out.with_suffix('.zip'),allow_scripted=True)
        assert len(rows)==2 and not live
        if preset=='baseline1':assert not any(role=='agent' for role,msg in calls)
        if preset=='gold_contract':assert not list((out/'contracts').glob('*.json'))
        paths.append(out.with_suffix('.zip'))
    report=collect(paths,tmp_path/'six',allow_scripted=True,six=True)
    assert set(report['summary'])==set(SYSTEMS)

@pytest.mark.parametrize('layout',['zip','extracted','both'])
def test_collector_discovers_kaggle_auto_extracted_results(tmp_path,layout):
    inputs=tmp_path/'input';inputs.mkdir();run_dir=tmp_path/'run';run_dir.mkdir()
    for method in ['selective','tools']:
        data=json.dumps(dict(experiment='stage2-dual-parser-3b-fp16-v1',method=method,preset='proposed')).encode()
        seal=dict(complete=True,sha256={'plan.json':hashlib.sha256(data).hexdigest()})
        if layout in ['zip','both']:
            with zipfile.ZipFile(inputs/f'stage2_dual_3b_{method}_proposed.zip','w') as z:
                z.writestr('plan.json',data);z.writestr('bundle.json',json.dumps(seal))
        if layout in ['extracted','both']:
            folder=inputs/method;folder.mkdir();(folder/'plan.json').write_bytes(data);(folder/'bundle.json').write_text(json.dumps(seal))
    book=nbformat.read(ROOT/'notebooks/kaggle/stage2_dual_3b/stage2_dual_3b_collect_kaggle.ipynb',as_version=4)
    code=book.cells[3].source.split("OUT=RUN_DIR/")[0]
    namespace=dict(INPUT=inputs,RUN_DIR=run_dir,Path=Path,zipfile=zipfile,json=json,hashlib=hashlib)
    exec(code,namespace)
    assert len(namespace['paths'])==2
    for path in namespace['paths']:
        with zipfile.ZipFile(path) as z:assert z.testzip() is None


def test_generator_and_repair_use_the_same_pinned_unquantized_model():
    from eval.stage2_dual_3b.config import MODELS,EXPERIMENT
    assert MODELS['agent']==MODELS['generator']
    assert MODELS['agent']['id']=='Qwen/Qwen2.5-3B-Instruct'
    assert MODELS['agent']['four_bit'] is False
    assert MODELS['parser']['four_bit'] is True
    assert EXPERIMENT=='stage2-dual-parser-3b-fp16-v1'

@pytest.mark.parametrize('role',['generator','agent'])
def test_real_backend_load_arguments_do_not_quantize_3b(monkeypatch,role):
    import types
    from eval.stage2_dual_3b.backend import Backend
    loaded=[]
    class Parameter:
        device='cuda:0';dtype='float16'
        def is_floating_point(self):return True
    class Model:
        def parameters(self):return iter([Parameter()])
        def eval(self):return self
    class Models:
        @staticmethod
        def from_pretrained(name,**kwargs):loaded.append((name,kwargs));return Model()
    class Tokenizers:
        @staticmethod
        def from_pretrained(*args,**kwargs):return types.SimpleNamespace(pad_token='pad',eos_token='eos')
    def quantization_forbidden(*a,**kw):raise AssertionError('Quantization configured for generator/repair')
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(float16='float16',cuda=types.SimpleNamespace(device_count=lambda:1)))
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoModelForCausalLM=Models,AutoTokenizer=Tokenizers,BitsAndBytesConfig=quantization_forbidden))
    backend=Backend(role,'selective')
    assert backend.precision['quantization']=='none' and backend.precision['compute_dtype']=='torch.float16'
    assert loaded[0][0]=='Qwen/Qwen2.5-3B-Instruct'
    assert loaded[0][1]['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert loaded[0][1]['dtype']=='float16' and 'quantization_config' not in loaded[0][1]


def test_quantized_or_wrong_dtype_repair_weights_are_rejected(monkeypatch):
    import types
    from eval.stage2_dual_3b.backend import Backend
    class Parameter:
        device='cuda:0';dtype='float32'
        def is_floating_point(self):return True
    class Model:
        is_loaded_in_4bit=False
        def parameters(self):return iter([Parameter()])
        def eval(self):return self
    model=Model()
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(float16='float16',cuda=types.SimpleNamespace(device_count=lambda:1)))
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a,**kw:model),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:types.SimpleNamespace(pad_token='pad')),BitsAndBytesConfig=lambda **kw:None))
    with pytest.raises(RuntimeError,match='FP16'):Backend('agent','tools')
    model.is_loaded_in_4bit=True
    with pytest.raises(RuntimeError,match='unquantized'):Backend('agent','tools')


def test_collector_rejects_seven_billion_model_configuration(tmp_path):
    factory,*_=pipeline_factory('selective')
    out=run('selective',tmp_path/'source',smoke=True,backend_factory=factory)
    plan=json.loads((out/'plan.json').read_text())
    plan['models']['agent']=dict(id='Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',four_bit=True)
    (out/'plan.json').write_text(json.dumps(plan));export(out,True)
    with pytest.raises(ValueError,match='settings'):read_bundle(out.with_suffix('.zip'),allow_scripted=True)
