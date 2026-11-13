"""Independence and live parser integration; scripted executions are never quality results."""
import json
from pathlib import Path
import pandas as pd
import pytest
from igt.llm import ScriptedBackend
from igt.agents.intent_parser import contract_to_text
from eval.stage2_independent.context import parser_messages
from eval.stage2_independent.runner import run,load_tasks,execute
from eval.stage2_independent.collect import read_bundle,collect
from eval.stage2_independent.config import SYSTEMS


def test_profile_only_uses_public_input():
    task=load_tasks()[0][1];root=Path(__file__).resolve().parents[1];df=pd.read_csv(root/task['data_file'])
    actual=parser_messages(task,df)
    assert parser_messages(dict(task,reference_code='SECRET',gold_contract={'SECRET':1}),df)==actual
    assert 'median' in actual[-1]['content']
    assert 'SECRET' not in str(actual)
    for p,t in load_tasks():
        # All current dataset/request profiles fit before any real GPU inference.
        assert len(parser_messages(t,pd.read_csv(root/t['data_file']))[-1]['content'])<22000


def test_invalid_live_contract_blocks_generation():
    task=load_tasks()[0][1]
    generator=ScriptedBackend([])
    result=execute(task,pd.DataFrame({'x':[1]}),'proposed',dict(extracted=dict(status='invalid_contract',contract=None),tokens=5,llm_calls=1),generator,generator)
    assert result['record']['outcome']=='ESCALATE'
    assert result['record']['tokens']==5 and not generator.calls


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    root=tmp_path_factory.mktemp('independent-stage2');task=load_tasks()[0][1]
    calls=[]
    def factory(role):
        calls.append(role)
        if role=='parser':return ScriptedBackend([contract_to_text(task['gold_contract'])],model='scripted-parser')
        if role=='generator':return ScriptedBackend(['```python\n'+task['reference_code']+'```'],model='scripted-generator')
        # Baseline3 freshly asks this backend for a contract, unloads it, then loads a new repair backend.
        return ScriptedBackend([contract_to_text(task['gold_contract'])]+['{"action":"escalate","reason":"scripted diagnostic"}']*10,model='scripted-agent')
    for system in SYSTEMS:run(system,root/system,limit=1,backend_factory=factory)
    return root,calls


def test_fresh_six_arm_workflow_and_strict_collector(complete):
    root,calls=complete
    paths=[(root/s).with_suffix('.zip') for s in SYSTEMS]
    for p in paths:
        plan,rows=read_bundle(p,allow_scripted=True)
        assert len(rows)==1
        assert rows[0]['tokens']>=rows[0]['parser_tokens']
        with pytest.raises(ValueError,match='real-model'):read_bundle(p)
    result=collect(paths,root/'collected',allow_scripted=True)
    assert result['model_quality_result'] is False
    assert result['summary']['proposed']['correct']==1
    assert calls.count('parser')==2
    parsed=json.loads(next((root/'proposed/contracts').glob('*.json')).read_text())
    assert parsed['extracted']['status']=='ok' and 'profile' in parsed['messages'][-1]['content']
    # No phase consumes another run's files or an earlier project's parse/threshold result.
    plan=json.loads((root/'proposed/plan.json').read_text())
    assert not any('results/' in k or 'kaggle_outputs/' in k for k in plan['inputs'])
    with pytest.raises(ValueError,match='empty'):run('proposed',root/'proposed',backend_factory=lambda role:None)


def test_pack_contains_only_raw_inputs_and_code(tmp_path):
    import ast,base64,zlib,nbformat,subprocess
    root=Path(__file__).resolve().parents[1]
    folder=root/'notebooks/kaggle/stage2_independent'
    for path in folder.glob('*.ipynb'):
        book=nbformat.read(path,as_version=4);nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type=='code':compile(cell.source,str(path),'exec')
        literals={}
        for node in ast.parse(book.cells[1].source).body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['PAYLOAD','PAYLOAD_SHA']:
                literals[node.targets[0].id]=ast.literal_eval(node.value)
        import hashlib
        raw=zlib.decompress(base64.b64decode(literals['PAYLOAD']))
        assert hashlib.sha256(raw).hexdigest()==literals['PAYLOAD_SHA']
        files=json.loads(raw)
        assert not any(n.startswith(('results/','kaggle_outputs/','benchmark/pool/','finetune/runs/')) for n in files)
        for name,encoded_source in files.items():
            source=base64.b64decode(encoded_source)
            assert (root/name).read_bytes()==source
            target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source)
    # A fresh isolated source tree has no earlier experiment outputs and imports the whole workflow.
    result=subprocess.run([__import__('sys').executable,'-m','eval.stage2_independent.runner','--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not (tmp_path/'results').exists() and not (tmp_path/'kaggle_outputs').exists()


def test_parser_unloads_before_generation(tmp_path):
    task=load_tasks()[0][1];events=[]
    class Script(ScriptedBackend):
        def __init__(self,role,replies):
            super().__init__(replies,model=role);self.role=role;events.append('load '+role)
        def close(self):events.append('close '+self.role)
    def factory(role):
        replies=[contract_to_text(task['gold_contract'])] if role=='parser' else ['```python\n'+task['reference_code']+'```'] if role=='generator' else []
        return Script(role,replies)
    run('proposed',tmp_path/'proposed',limit=1,backend_factory=factory)
    assert events.index('close parser')<events.index('load generator')


@pytest.mark.parametrize('damage',['incomplete','missing_contract','changed_settings','changed_raw_input'])
def test_collector_rejects_inconsistent_runs(complete,tmp_path,damage):
    import hashlib,zipfile
    root,_=complete
    with zipfile.ZipFile((root/'proposed').with_suffix('.zip')) as z:
        files={n:z.read(n) for n in z.namelist()}
    seal=json.loads(files.pop('bundle.json'))
    if damage=='incomplete':seal['complete']=False
    elif damage=='missing_contract':files.pop(next(n for n in files if n.startswith('contracts/')))
    else:
        plan=json.loads(files['plan.json'])
        if damage=='changed_settings':plan['settings']['seed']=123
        else:plan['inputs'][next(iter(plan['inputs']))]='tampered'
        files['plan.json']=json.dumps(plan).encode()
    seal['sha256']={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    target=tmp_path/'damaged.zip'
    with zipfile.ZipFile(target,'w') as z:
        for n,b in files.items():z.writestr(n,b)
        z.writestr('bundle.json',json.dumps(seal))
    with pytest.raises(ValueError):read_bundle(target,allow_scripted=True)


def test_incompatible_optional_torchao_is_removed_before_model_loading():
    import importlib.metadata as metadata
    from eval.stage2_independent.environment import prepare_optional_packages
    installed={'torchao':'0.10.0'};commands=[]
    def version(name):
        if name not in installed:raise metadata.PackageNotFoundError(name)
        return installed[name]
    def run(command,check):
        assert check is True
        commands.append(command);installed.pop('torchao')
    result=prepare_optional_packages(version,run,executable='/kaggle/python')
    assert result['action']=='removed_incompatible'
    assert commands==[['/kaggle/python','-m','pip','uninstall','-y','torchao']]
    assert prepare_optional_packages(version,run)['action']=='already_absent'
    installed['torchao']='0.16.0'
    assert prepare_optional_packages(version,run)['action']=='compatible_unchanged'
    assert len(commands)==1


def test_setup_fails_if_incompatible_package_remains():
    from eval.stage2_independent.environment import prepare_optional_packages
    with pytest.raises(RuntimeError,match='still installed'):
        prepare_optional_packages(lambda name:'0.10.0',lambda command,check:None)
