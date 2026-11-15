"""End-to-end combined integration, version isolation and notebook inputs."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
import nbformat
import pytest
from igt.agents.intent_parser import contract_to_text
from eval.stage2_combined_3b.runner import run,load_tasks,SMOKE_IDS
from eval.stage2_combined_3b.collect import read_bundle,collect
from eval.stage2_combined_3b.artifacts import export
from eval.stage2_combined_3b.config import EXPERIMENT,REPAIR_VERSION
ROOT=Path(__file__).resolve().parents[1]


def factory(method,invalid=False):
    tasks={t['id']:t for _,t in load_tasks() if t['id'] in SMOKE_IDS}
    live=[];calls=[];lifecycle=[];tool_steps={}
    class Backend:
        def __init__(self,role):
            assert not live
            self.role=role;self.model='scripted-'+role;live.append(self);lifecycle.append(('load',role))
        def prompt_tokens(self,messages):return 0
        def complete(self,messages,*args,**kwargs):
            calls.append((self.role,copy.deepcopy(messages)))
            t=next(t for t in tasks.values() if ' '.join(t['request'].split()) in ' '.join(messages[1]['content'].split()))
            if self.role=='parser':
                if invalid:return 'invalid',100,50
                if method=='tools' and t['id']=='titanic_miss_004':
                    step=tool_steps.get(t['id'],0);tool_steps[t['id']]=step+1
                    if step==0:return '{"tool":"profile_column","args":{"column":"Age"}}',100,50
                return contract_to_text(t['gold_contract']),100,50
            if self.role=='generator':
                tree=ast.parse(t['reference_code'])
                class AddHelper(ast.NodeTransformer):
                    def visit_Return(self,node):
                        node.value=ast.Call(func=ast.Attribute(value=node.value,attr='assign',ctx=ast.Load()),args=[],keywords=[ast.keyword(arg='unrequested_helper',value=ast.Constant(1))]);return node
                tree=AddHelper().visit(tree);ast.fix_missing_locations(tree)
                return '```python\n'+ast.unparse(tree)+'\n```',100,50
            return json.dumps(dict(action='propose_fix',code=t['reference_code'])),100,50
        def close(self):
            assert live==[self];live.clear();lifecycle.append(('close',self.role))
    return Backend,live,calls,lifecycle


@pytest.mark.parametrize('method',['selective','tools'])
def test_fresh_pipeline_guard_repairs_and_records_cost(method,tmp_path):
    backend,live,calls,lifecycle=factory(method)
    out=run(method,tmp_path/method,smoke=True,backend_factory=backend)
    plan,rows=read_bundle(out.with_suffix('.zip'),allow_scripted=True)
    assert plan['experiment']==EXPERIMENT and plan['repair_prompt']==REPAIR_VERSION
    assert plan['common']['repair_arm']=='combined'
    assert all(r['final_correct'] and not r['initial_correct'] for r in rows)
    assert all(r['attempts']==1 and r['repair_diagnostic_seconds']>=0 and r['repair_arm']=='combined' for r in rows)
    assert all(r['combined_reference_verdict']['status']=='PASS' for r in rows)
    assert not live and lifecycle==[(op,role) for role in ['parser','generator','agent'] for op in ['load','close']]
    assert len([1 for role,_ in calls if role=='generator'])==2
    assert len([1 for role,_ in calls if role=='agent'])==2
    for role,messages in calls:
        assert 'reference_code' not in str(messages) and 'gold_contract' not in str(messages)
        if role=='agent':assert 'declared_output_schema' in str(messages)
        if role=='parser':assert 'FRESH RECOVERY' not in str(messages)
    summary=json.loads((out/'summary.json').read_text())
    assert summary['summary']['correct']==2 and not summary['model_quality_result']
    with pytest.raises(ValueError,match='real-model'):read_bundle(out.with_suffix('.zip'))
    with pytest.raises(ValueError,match='fresh output'):run(method,out,smoke=True,backend_factory=backend)


def test_invalid_contract_skips_models_and_guard(tmp_path):
    backend,live,calls,life=factory('selective',invalid=True)
    out=run('selective',tmp_path/'invalid',smoke=True,backend_factory=backend)
    _,rows=read_bundle(out.with_suffix('.zip'),allow_scripted=True)
    assert all(r['stage_ended']=='parse' and r['outcome']=='ESCALATE' for r in rows)
    assert life==[('load','parser'),('close','parser')]
    assert not list((out/'generation').glob('*.json'))


def test_collector_accepts_two_methods_and_rejects_changed_version(tmp_path):
    paths=[]
    for method in ['selective','tools']:
        backend,*_=factory(method)
        out=run(method,tmp_path/method,smoke=True,backend_factory=backend);paths.append(out.with_suffix('.zip'))
    summary=collect(paths,tmp_path/'comparison',allow_scripted=True)
    assert all(s['correct']==2 for s in summary['summary'].values())
    assert not summary['model_quality_result']
    with pytest.raises(ValueError,match='each method'):collect([paths[0],paths[0]],tmp_path/'dup',allow_scripted=True)
    out=tmp_path/'selective';plan=json.loads((out/'plan.json').read_text());plan['experiment']='stage2-dual-parser-3b-fp16-v1'
    (out/'plan.json').write_text(json.dumps(plan));export(out,True)
    with pytest.raises(ValueError,match='settings'):read_bundle(paths[0],allow_scripted=True)


@pytest.mark.parametrize('packed',[True,False])
@pytest.mark.parametrize('method',['selective','tools'])
def test_notebook_source_discovery_and_language(packed,method,tmp_path,monkeypatch):
    folder=ROOT/'notebooks/kaggle/stage2_combined_3b';source=folder/'stage2_combined_3b_source.zip'
    inputs=tmp_path/'input';work=tmp_path/'work';inputs.mkdir();work.mkdir()
    if packed:shutil.copy(source,inputs/source.name)
    else:
        with zipfile.ZipFile(source) as z:z.extractall(inputs)
    book=nbformat.read(folder/f'stage2_combined_3b_{method}_proposed_kaggle.ipynb',as_version=4)
    assert book.metadata.language_info.name=='python'
    for c in book.cells:
        if c.cell_type=='code':compile(c.source,'notebook','exec')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:None)
    ns={};exec(book.cells[1].source.replace("INPUT=Path('/kaggle/input')",f'INPUT=Path({str(inputs)!r})').replace("WORK=Path('/kaggle/working')",f'WORK=Path({str(work)!r})'),ns)
    exec(book.cells[2].source,ns)
    manifest=json.loads((ns['ROOT']/'_stage2_combined_3b_manifest.json').read_text())
    assert manifest['experiment']==EXPERIMENT
    assert all(hashlib.sha256((ns['ROOT']/p).read_bytes()).hexdigest()==sha for p,sha in manifest['sha256'].items())
    assert 'split=\'dev\'' in book.cells[4].source
    assert 'eval.stage2_combined_3b.runner' in book.cells[4].source


def test_selected_parsers_and_models_match_control():
    from eval.stage2_dual_3b import config as old
    from eval.stage2_combined_3b import config as new
    from eval.stage2_dual_3b.parsers import parse as old_parse
    from eval.stage2_combined_3b.parsers import parse as new_parse
    assert old_parse is new_parse
    assert new.MODELS==old.MODELS and new.PARSER_SETTINGS==old.PARSER_SETTINGS and new.PROTOCOLS==old.PROTOCOLS
    assert new.MODELS['agent']['four_bit'] is False
    assert new.MODELS['generator']==new.MODELS['agent']
