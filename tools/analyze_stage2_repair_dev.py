"""Audit the sealed live repair development run and replay saved replies on CPU."""
from pathlib import Path
import dataclasses
import hashlib
import json
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import nbformat
from igt.llm import LLMClient,ScriptedBackend
from igt.agents.repair_agent import AgentConfig
from igt.orchestrator import Limits
from igt.sandbox import safe_run
from eval.equivalence import outputs_equal
from igt.verifier import verify
from eval.stage2_repair_dev.agent import run_repair,ARMS,VERSION
from eval.stage2_repair_dev.suite import build_suite,VERSION as SUITE_VERSION
from eval.stage2_repair_dev.runner import check_case,summarize,EXPERIMENT
from eval.stage2_repair_dev.artifacts import runtime_hashes

OUT=ROOT/'results/stage2_repair_dev_analysis/v1'
BUNDLE=ROOT/'kaggle_outputs/stage2/repair_dev_v1/stage2_repair_dev_full.zip'
BOOK=ROOT/'notebooks/executed/stage2/repair_dev_v1/stage2_repair_dev_kaggle_executed.ipynb'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def normalized_trace(trace):
    return [{k:v for k,v in e.items() if k not in ['seconds','tokens']} for e in trace]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(BUNDLE) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        seal=json.loads(z.read('bundle.json'));assert seal['complete']
        assert set(seal['sha256'])==set(z.namelist())-{'bundle.json'}
        for name,sha in seal['sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==sha,name
        payload={n:json.loads(z.read(n)) for n in z.namelist() if n.endswith('.json') and n!='bundle.json'}
    plan=payload['plan.json'];summary=payload['summary.json'];completion=payload['completion.json']
    assert plan['experiment']==EXPERIMENT and plan['repair_version']==VERSION and plan['suite']==SUITE_VERSION
    assert not plan['smoke'] and plan['model_quality_result'] and summary['model_quality_result']
    assert completion==dict(complete=True,records=68,expected=68,model_quality_result=True)
    assert plan['runtime']==runtime_hashes()
    source=ROOT/'notebooks/kaggle/stage2_repair_dev/stage2_repair_dev_source.zip'
    with zipfile.ZipFile(source) as z:
        raw=z.read('stage2_repair_dev_source/manifest.json');assert hashlib.sha256(raw).hexdigest()==plan['source_manifest_sha256']
        manifest=json.loads(raw)
        for name,sha in manifest['sha256'].items():assert digest(ROOT/name)==sha
    assert plan['model']['four_bit'] is False
    assert plan['resolved_model']['precision']==dict(quantization='none',compute_dtype='torch.float16',floating_parameter_dtypes=['torch.float16'])
    assert plan['resolved_model']['model']==plan['model']['id'] and plan['resolved_model']['revision']==plan['model']['revision']
    assert plan['limits']==Limits().to_dict()
    cases=build_suite();assert hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()==plan['suite_sha256']
    assert plan['task_ids']==[c['id'] for c in cases] and plan['arms']==list(ARMS)
    records={}
    for name,value in payload.items():
        if name.startswith('tasks/'):
            key=(value['record']['task_id'],value['record']['arm']);assert key not in records;records[key]=value
    assert set(records)=={(c['id'],a) for c in cases for a in ARMS}
    computed=summarize([v['record'] for v in records.values()],tuple(ARMS));assert computed==summary['arms']
    book=nbformat.read(BOOK,as_version=4);template=nbformat.read(ROOT/'notebooks/kaggle/stage2_repair_dev/stage2_repair_dev_kaggle.ipynb',as_version=4)
    assert [c.source for c in book.cells]==[c.source for c in template.cells]
    errors=[o.get('evalue') for c in book.cells for o in c.get('outputs',[]) if o.output_type=='error'];assert not errors
    audit=dict(bundle_sha256=digest(BUNDLE),notebook_sha256=digest(BOOK),source_manifest_sha256=plan['source_manifest_sha256'],
        model=plan['resolved_model'],environment=plan['environment'],summary=computed,checks=dict(seal=True,runtime=True,source=True,coverage=True,summary=True,notebook_source=True,no_notebook_errors=True),
        mismatches=[],cases={},limits=['Local CPU replay; GPU latency and real token counts are verified from recorded traces, not recomputed by a tokenizer.',
        'Small synthetic repair-only suite informed by earlier failures; no live parser/generator and no independent benchmark.',
        'Five correct controls return before any repair model call; this does not test regression safety when correct code is blocked by an incorrect contract.'])
    for c in cases:
        before,reference,initial,initial_correct=check_case(c)
        details={}
        for arm in ARMS:
            saved=records[c['id'],arm];row=saved['record'];trace=saved['trace']
            assert saved['request']==c['request'] and saved['contract']==c['contract'] and saved['initial_code']==c['initial_code']
            assert row['initial_correct']==initial_correct
            assert row['tokens']==sum(e['tokens'] for e in trace if e['event']=='llm')
            assert row['llm_calls']==sum(e['event']=='llm' for e in trace)
            assert row['tool_calls']==sum(e['event']=='tool' for e in trace)
            assert row['fresh_recovery_requests']==sum(e['event']=='fresh_recovery_requested' for e in trace)
            backend=ScriptedBackend([e['reply'] for e in trace if e['event']=='llm'])
            result=run_repair(dict(request=c['request']),c['initial_code'],before,lambda after:verify(c['contract'],before,after)[0],
                LLMClient(backend),AgentConfig(limits=Limits()),contract_text=json.dumps(c['contract']),
                schema_text=json.dumps(dict(columns={name:str(before[name].dtype) for name in before},rows=len(before))),arm=arm)
            correct,reason=outputs_equal(reference,result.final_output) if result.outcome=='ACCEPT' else (False,'withheld')
            expected=dict(outcome=row['outcome'],attempts=row['attempts'],final_code=saved['final_code'],final_correct=row['final_correct'])
            actual=dict(outcome=result.outcome,attempts=result.attempts,final_code=result.final_code,final_correct=bool(correct))
            if expected!=actual:audit['mismatches'].append(dict(task=c['id'],arm=arm,kind='result',expected=expected,actual=actual))
            if normalized_trace(trace)!=normalized_trace(result.trace):audit['mismatches'].append(dict(task=c['id'],arm=arm,kind='trace'))
            details[arm]=dict(record=row,final_code=saved['final_code'],equivalence=reason,
                recovery_after_steps=[e['step'] for e in trace if e['event']=='fresh_recovery_requested'])
        audit['cases'][c['id']]=dict(request=c['request'],initial_correct=initial_correct,arms=details)
        print('CPU replay:',c['id'],flush=True)
    control={c['id']:records[c['id'],'control']['record'] for c in cases}
    audit['paired']={arm:dict(gains=[c['id'] for c in cases if records[c['id'],arm]['record']['final_correct'] and not control[c['id']]['final_correct']],
                              losses=[c['id'] for c in cases if control[c['id']]['final_correct'] and not records[c['id'],arm]['record']['final_correct']]) for arm in ARMS}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    assert not audit['mismatches'],audit['mismatches']
    print('PASSED: sealed live run; all 68 CPU replays match outcomes, final code and traces.')
if __name__=='__main__':main()
