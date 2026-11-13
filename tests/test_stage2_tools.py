import json
from pathlib import Path
import pandas as pd
import pytest
from igt.llm import ScriptedBackend
from igt.agents.intent_parser import contract_to_text
from eval.stage2_tools.data_tools import DataTools
from eval.stage2_tools.parser import parse_fresh
from eval.stage2_tools.context import parser_messages
from eval.stage2_tools.runner import run,load_tasks
from eval.stage2_tools.collect import collect,read_bundle
from eval.stage2_tools.config import SYSTEMS

TASK=dict(id='test',request='Fill missing value with the median in region A.',data_file='input.csv')
GOLD=dict(contract_version='1.0',must_change=[],must_preserve=[])

def frame():return pd.DataFrame({'value':[1.123456789,3.987654321,None,99.0], 'region':['A','A','A','B']})

def test_tools_precision_conditions_groups_and_read_only():
    df=frame();before=df.copy(deep=True);tools=DataTools(df)
    r=tools.call('profile_column',dict(column='value',where=dict(column='region',op='==',value='A')))
    assert r['ok'] and r['result']['statistics']['median']==pytest.approx(2.555555555,abs=1e-12)
    assert r['result']['missing']==1
    assert tools.call('count_rows',dict(where=dict(column='value',op='is_null')))['result']['matching_rows']==1
    groups=tools.call('group_profile',dict(column='value',by=['region']))
    assert groups['ok'] and len(groups['result']['groups'])==2
    assert tools.call('inspect_rows',dict(columns=['value'],n=1))['result']['rows'][0]['value']==1.123456789
    pd.testing.assert_frame_equal(df,before)
    tie=DataTools(pd.DataFrame({'x':['a','b',None]})).call('profile_column',dict(column='x'))['result']
    assert tie['mode_tie'] and tie['modes']==['a','b']
    missing=tools.call('profile_column',dict(column='value',where=dict(column='region',op='==',value='absent')))
    assert missing['ok'] and missing['result']['statistics']['median'] is None


def test_tool_errors_are_explicit_and_json_complete():
    tools=DataTools(frame())
    for name,args in [('run_python',{'code':'print(1)'}),('profile_column',{'column':'absent'}),('inspect_rows',{'columns':['value'],'n':100}),('count_rows',{'where':{'column':'absent','op':'is_null'}})]:
        response=tools.call(name,args);assert not response['ok'];json.loads(json.dumps(response))
    big=DataTools(pd.DataFrame({'x':['x'*7000]})).call('profile_column',dict(column='x'))
    assert not big['ok'] and 'exceeds' in big['error']
    groups=DataTools(pd.DataFrame({'x':list(range(13)),'g':list(range(13))})).call('group_profile',dict(column='x',by=['g']))
    assert not groups['ok']


def test_model_requests_tool_observes_result_then_contract():
    def final(messages):
        observation=json.loads(messages[-1]['content'].splitlines()[1])
        assert observation['result']['statistics']['median']==(1.123456789+3.987654321)/2
        return contract_to_text(GOLD)
    b=ScriptedBackend([json.dumps(dict(tool='profile_column',args=dict(column='value',where=dict(column='region',op='==',value='A')))),final])
    r=parse_fresh(TASK,frame(),b,'p1-v2')
    assert r['extracted']['status']=='ok' and r['tool_attempts']==1 and r['llm_calls']==2
    assert r['tokens']>0 and r['tool_failures']==0
    assert len(r['messages'])==5


def test_no_tool_needed_and_no_gold_leak():
    msg=parser_messages(TASK,frame())
    assert msg==parser_messages(dict(TASK,gold_contract={'secret':1},reference_code='SECRET'),frame())
    assert 'Input dataset profile' not in msg[-1]['content']
    b=ScriptedBackend([contract_to_text(GOLD)])
    r=parse_fresh(TASK,frame(),b,'p1-v2')
    assert r['tool_attempts']==0 and r['extracted']['status']=='ok'


def test_error_recovery_and_escalation():
    b=ScriptedBackend(['{"tool":"profile_column","args":{"column":"absent"}}','{"action":"escalate","reason":"Requested column absent"}'])
    r=parse_fresh(TASK,frame(),b,'p1-v2')
    assert r['extracted']['status']=='parser_escalation' and r['tool_failures']==1
    assert not r['extracted']['contract']


@pytest.mark.parametrize('case',['tool','invalid','calls','tokens','context'])
def test_limits_are_explicit_withheld_outcomes(case):
    from eval.stage2_tools.backend import ContextLimitError
    kwargs={}
    if case=='tool':replies=['{"tool":"profile_column","args":{"column":"value"}}']*2;kwargs={'max_tools':1};expected='parser_tool_limit'
    elif case=='invalid':replies=['bad']*3;expected='parser_invalid_response'
    elif case=='calls':replies=['{"tool":"profile_column","args":{"column":"value"}}'];kwargs={'max_calls':1};expected='parser_call_limit'
    elif case=='tokens':replies=[];kwargs={'token_budget':1};expected='parser_token_limit'
    else:replies=[];expected='parser_context_limit'
    b=ScriptedBackend(replies)
    if case=='context':
        def count(messages):raise ContextLimitError('too long')
        b.prompt_tokens=count
    r=parse_fresh(TASK,frame(),b,'p1-v2',**kwargs)
    assert r['extracted']['status']==expected and r['extracted']['contract'] is None


def test_gpu_infrastructure_errors_are_not_hidden_as_parser_outcomes():
    b=ScriptedBackend([])
    def fail(messages):raise RuntimeError('CUDA out of memory')
    b.prompt_tokens=fail
    with pytest.raises(RuntimeError,match='CUDA'):parse_fresh(TASK,frame(),b,'p1-v2')


def test_all_six_routes_generate_their_own_tool_contracts(tmp_path):
    task=load_tasks()[0][1];root=Path(__file__).resolve().parents[1]
    df=pd.read_csv(root/task['data_file']);column=next(c for c in df if pd.api.types.is_numeric_dtype(df[c]))
    query=json.dumps(dict(tool='profile_column',args=dict(column=column)))
    def factory(role):
        if role=='generator':return ScriptedBackend(['```python\n'+task['reference_code']+'```'],model='scripted-generator')
        return ScriptedBackend([query,contract_to_text(task['gold_contract'])]+['{"action":"escalate","reason":"test"}']*10,model='scripted-parser-or-repair')
    for system in SYSTEMS:run(system,tmp_path/system,limit=1,backend_factory=factory)
    paths=[(tmp_path/s).with_suffix('.zip') for s in SYSTEMS]
    report=collect(paths,tmp_path/'summary',allow_scripted=True)
    assert report['summary']['proposed']['correct']==1
    assert report['summary']['proposed']['parser_tool_calls']==1
    assert report['summary']['proposed']['parser_tool_failures']==0
    with pytest.raises(ValueError,match='real-model'):read_bundle(paths[0])


def test_pack_and_profile_version_unchanged(tmp_path):
    import ast,base64,zlib,hashlib,nbformat,subprocess,sys
    root=Path(__file__).resolve().parents[1]
    for p in (root/'notebooks/kaggle/stage2_tools').glob('*.ipynb'):
        b=nbformat.read(p,as_version=4);nbformat.validate(b)
        for c in b.cells:
            if c.cell_type=='code':compile(c.source,str(p),'exec')
        literals={}
        for node in ast.parse(b.cells[1].source).body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['PAYLOAD','PAYLOAD_SHA']:literals[node.targets[0].id]=ast.literal_eval(node.value)
        raw=zlib.decompress(base64.b64decode(literals['PAYLOAD']));assert hashlib.sha256(raw).hexdigest()==literals['PAYLOAD_SHA']
        for name,data in json.loads(raw).items():
            data=base64.b64decode(data);assert data==(root/name).read_bytes()
            assert not name.startswith(('results/','kaggle_outputs/','eval/stage2_independent/'))
            dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    result=subprocess.run([sys.executable,'-m','eval.stage2_tools.runner','--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
