"""Behavioral tests for bounded recovery, trustworthy evidence and closed output schemas."""
import copy
import json
import pandas as pd
import pytest
from igt.agents.repair_agent import AgentConfig, run_repair as old_run
from igt.llm import LLMClient, ScriptedBackend
from igt.orchestrator import Limits
from igt.verifier import verify
from igt.pipeline.failure_report import failure_report
from eval.stage2_repair_dev.agent import run_repair
from eval.stage2_repair_dev.guard import guarded_verifier, schema_failures
from eval.stage2_repair_dev.evidence import diagnose
from eval.stage2_repair_dev.suite import build_suite, frame
from eval.stage2_repair_dev.runner import run


def case(name):
    return next(c for c in build_suite() if c['id']==name)


def fix(code):
    return json.dumps(dict(action='propose_fix',code=code))


def tool(name,**args):
    return json.dumps(dict(action='tool',tool=name,args=args))


def execute(c,replies,arm='combined',limits=None):
    before=frame(c);backend=ScriptedBackend(replies)
    result=run_repair(dict(request=c['request'],reference_code='SECRET_REFERENCE',gold_contract='SECRET_GOLD'),
        c['initial_code'],before,lambda out:verify(c['contract'],before,out)[0],LLMClient(backend),
        AgentConfig(limits=limits or Limits()),contract_text=json.dumps(c['contract']),schema_text=str(before.dtypes),arm=arm)
    return result,backend


def test_control_matches_original_prompts_trace_and_result():
    c=case('category_assignment_1');before=frame(c)
    result,backend=execute(c,[fix(c['reference_code'])],arm='control')
    other=ScriptedBackend([fix(c['reference_code'])])
    original=old_run(dict(request=c['request']),c['initial_code'],before,lambda out:verify(c['contract'],before,out)[0],
        LLMClient(other),AgentConfig(limits=Limits()),contract_text=json.dumps(c['contract']),schema_text=str(before.dtypes))
    assert backend.calls==other.calls
    assert result.trace==original.trace and result.outcome==original.outcome=='ACCEPT'


def test_fresh_recovery_once_and_no_reference_leak():
    c=case('category_assignment_1')
    result,backend=execute(c,[fix(c['initial_code']),fix(c['reference_code'])])
    assert result.outcome=='ACCEPT' and result.attempts==2
    assert sum(e['event']=='fresh_recovery_requested' for e in result.trace)==1
    assert 'FRESH RECOVERY' in backend.calls[1][1]['content']
    assert 'observed_ordered' in backend.calls[0][1]['content']
    assert 'SECRET' not in str(backend.calls)


def test_recovery_never_extends_attempt_limit():
    c=case('category_assignment_1')
    result,backend=execute(c,[fix(c['initial_code'])]*5,limits=Limits(max_attempts=1))
    assert result.outcome=='REJECT' and result.attempts==1 and len(backend.calls)==1
    assert not any(e['event']=='fresh_recovery_requested' for e in result.trace)


def test_repeated_recovery_remains_bounded():
    c=case('category_assignment_1')
    result,backend=execute(c,[fix(c['initial_code'])]*8)
    assert result.outcome=='ESCALATE' and len(backend.calls)==2
    assert sum(e['event']=='fresh_recovery_requested' for e in result.trace)==1


def test_budget_still_stops_repair():
    c=case('category_assignment_1')
    result,backend=execute(c,[fix(c['initial_code'])]*8,limits=Limits(token_budget=1))
    assert result.outcome=='REJECT' and len(backend.calls)==1
    assert any(e['event']=='budget_exceeded' for e in result.trace)


def test_crash_has_no_fabricated_current_output():
    c=case('dedup_crashed_candidate')
    result,backend=execute(c,[tool('inspect_dataset',which='last_output'),fix(c['reference_code'])])
    assert result.outcome=='ACCEPT'
    assert '"output_available": false' in backend.calls[0][1]['content']
    assert [e for e in result.trace if e['event']=='tool'][0]['ok'] is False


def test_failed_tool_run_clears_previous_output():
    c=case('category_assignment_1')
    bad="def transform(df):\n    return df['does_not_exist']\n"
    result,backend=execute(c,[tool('run_transformation',code=bad),tool('inspect_dataset',which='last_output'),fix(c['reference_code'])])
    assert result.outcome=='ACCEPT'
    events=[e for e in result.trace if e['event']=='tool_observation_provenance']
    assert not events[0]['output_available'] and events[0]['output_code_hash'] is None
    assert [e for e in result.trace if e['event']=='tool'][1]['ok'] is False


def test_schema_guard_blocks_extra_column_accepted_by_old_verifier():
    c=case('control_fill');before=frame(c);after=before.fillna({'quantity':4}).assign(helper=1)
    vf=lambda out:verify(c['contract'],before,out)[0]
    assert vf(after).status=='PASS'
    verdict=guarded_verifier(vf,c['contract'],before)(after)
    assert verdict.status=='VIOLATION' and verdict.failures[-1]['value']['unexpected']==['helper']


@pytest.mark.parametrize('name',['control_fill','control_add','control_drop','control_rename','control_empty'])
def test_guard_preserves_legitimate_already_correct_tasks(name):
    result,backend=execute(case(name),[])
    assert result.outcome=='ACCEPT' and result.attempts==0 and backend.calls==[]


def test_duplicate_columns_are_rejected():
    c=case('control_fill');before=frame(c);after=before.fillna({'quantity':4})
    after=pd.concat([after,after[['quantity']]],axis=1)
    failures=schema_failures(c['contract'],before,after)
    assert failures[0]['value']['duplicate']==['quantity']


def test_guard_does_not_hide_original_violation():
    c=case('control_fill');before=frame(c);bad=before.copy()
    vf=lambda out:verify(c['contract'],before,out)[0]
    result=guarded_verifier(vf,c['contract'],before)(bad)
    assert result.failures==vf(bad).failures and result.status=='VIOLATION'


def test_guard_repair_reexecutes_and_verifies_candidate():
    c=case('filter_extra_helper')
    old,_=execute(c,[],arm='control')
    assert old.outcome=='ACCEPT'
    new,backend=execute(c,[fix(c['reference_code'])],arm='schema_guard')
    assert new.outcome=='ACCEPT' and new.attempts==1
    assert 'declared_output_schema' in str(backend.calls)
    assert list(new.final_output.columns)==['record','stamp']


def test_invalid_contract_is_uncertain_not_pass():
    before=pd.DataFrame({'a':[1]})
    assert schema_failures({'invalid':True},before,before)[0]['kind']=='uncertain'


def test_evidence_is_aligned_after_reset_index():
    c=case('filter_keep_unknown');before=frame(c)
    after=before[before.measurement>=10].reset_index(drop=True)
    verdict=verify(c['contract'],before,after)[0]
    d=diagnose(c['contract'],before,after,failure_report('verify',verdict=verdict))
    assert d['alignment']['method']=='content'
    assert d['examples'] and all(not e['retained'] for e in d['examples'])
    assert {e['input_position'] for e in d['examples']}=={0,4}


def test_duplicate_input_indices_omit_ambiguous_examples():
    before=pd.DataFrame({'a':[1,2]},index=[0,0])
    d=diagnose({},before,before,dict(failures=[]))
    assert not d['examples'] and 'Ambiguous' in d['notes'][0]


def test_round_evidence_uses_contract_values_not_reference():
    c=case('rounding_convention_1');before=frame(c);after=before.copy()
    after['reading']=after.reading.apply(lambda x:round(x,1))
    v=verify(c['contract'],before,after)[0]
    d=diagnose(c['contract'],before,after,failure_report('verify',verdict=v))
    assert any(e['expected_from_contract']!=e['observed'] for e in d['examples'])
    assert 'reference_code' not in str(d)


def test_smoke_runner_is_labeled_infrastructure_and_exports_sealed_bundle(tmp_path):
    out=tmp_path/'run';summary=run(out,smoke=True,cpu_check=True)
    assert summary['complete'] and not summary['model_quality_result']
    assert summary['arms']['control']['incorrect_accepted']==1
    assert summary['arms']['combined']['incorrect_accepted']==0
    assert len(list((out/'tasks').glob('*.json')))==8 and out.with_suffix('.zip').exists()
    with pytest.raises(ValueError,match='fresh'):
        run(out,smoke=True,cpu_check=True)


def test_measurable_progress_does_not_trigger_fresh_reset():
    c=case('filter_keep_unknown')
    improved="def transform(df):\n    return df[(df['measurement'] >= 10) | (df['record'] == 1)].copy()\n"
    result,_=execute(c,[fix(improved),fix(c['reference_code'])])
    assert result.outcome=='ACCEPT' and result.attempts==2
    assert not any(e['event']=='fresh_recovery_requested' for e in result.trace)


@pytest.mark.parametrize('packed',[True,False])
def test_kaggle_loader_handles_zip_and_automatic_extraction(tmp_path,monkeypatch,packed):
    import hashlib
    import zipfile
    import shutil
    import subprocess
    import nbformat
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    source=root/'notebooks/kaggle/stage2_repair_dev/stage2_repair_dev_source.zip'
    inputs=tmp_path/'input';inputs.mkdir();work=tmp_path/'work';work.mkdir()
    if packed:
        shutil.copy(source,inputs/source.name)
    else:
        with zipfile.ZipFile(source) as z:z.extractall(inputs)
    book=nbformat.read(root/'notebooks/kaggle/stage2_repair_dev/stage2_repair_dev_kaggle.ipynb',as_version=4)
    assert book.metadata.language_info.name=='python'
    for cell in book.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook','exec')
    monkeypatch.setattr(subprocess,'run',lambda *args,**kwargs:None)
    namespace={};exec(book.cells[1].source.replace("INPUT=Path('/kaggle/input');WORK=Path('/kaggle/working')",f'INPUT=Path({str(inputs)!r});WORK=Path({str(work)!r})'),namespace)
    exec(book.cells[2].source,namespace)
    manifest=json.loads((namespace['ROOT']/'_repair_dev_manifest.json').read_text())
    assert all(hashlib.sha256((namespace['ROOT']/p).read_bytes()).hexdigest()==sha for p,sha in manifest['sha256'].items())
    assert not any(p.endswith(('.safetensors','.ipynb','.zip')) for p in manifest['sha256'])
    target=namespace['ROOT']/'eval/stage2_repair_dev/agent.py';target.write_text('modified')
    with pytest.raises(AssertionError,match='Changed working'):
        exec(book.cells[2].source,namespace)


def test_preservation_evidence_prioritizes_changed_column_beyond_first_eight():
    before=pd.DataFrame({f'column_{i}':[i,i+1] for i in range(12)})
    after=before.copy();after.loc[1,'column_11']=123
    contract=dict(must_change=[],must_preserve=[dict(type='columns_unchanged',target='*')])
    verdict=verify(contract,before,after)[0]
    d=diagnose(contract,before,after,failure_report('verify',verdict=verdict))
    assert d['examples'][0]['input']=={'column_11':12}
    assert d['examples'][0]['output']=={'column_11':123}
