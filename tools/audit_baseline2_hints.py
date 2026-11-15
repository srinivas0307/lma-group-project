"""CPU audit of the completed conventional-check baseline on the current repair stack."""
import collections,dataclasses,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import nbformat
import pandas as pd
from eval.stage2_selective.runner import load_tasks
from eval.stage2_baselines_hints.config import EXPERIMENT,CONVENTIONAL_THRESHOLDS,REPAIR_VERSIONS
from eval.stage2_baselines_hints.execute import conventional_checker
from eval.stage2_dual_3b.artifacts import digest,write
from eval.stage2_combined_3b.config import MODELS
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from igt.agents.code_generator_v2 import build_generator_messages,generator_prompt_info
from finetune.selective_facts_scopefix.context import schema_entry
from eval.equivalence import outputs_equal
OUT=ROOT/'results/baselines_hints_v1/baseline2_audit'

def load(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        raw={n:z.read(n) for n in z.namelist()}
    seal=json.loads(raw.pop('bundle.json'));assert seal['complete'] and seal['sha256']=={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}
    assert 'failure.json' not in raw
    return {n:json.loads(b) for n,b in raw.items() if n.endswith('.json')}

def main():
    path=ROOT/'stage2_baselines_hints_baseline2.zip'
    if not path.exists():path=ROOT/'kaggle_outputs/stage2/baselines_hints_v1/stage2_baselines_hints_baseline2.zip'
    b=load(path);plan=b['plan.json'];tasks=load_tasks('test')
    source=ROOT/'notebooks/kaggle/stage2_baselines_hints/stage2_baselines_hints_source.zip'
    with zipfile.ZipFile(source) as z:
        name=next(n for n in z.namelist() if n.endswith('/manifest.json') and n.count('/')==1);raw=z.read(name);prefix=name[:-len('manifest.json')]
        for n,sha in json.loads(raw)['sha256'].items():
            assert hashlib.sha256(z.read(prefix+n)).hexdigest()==sha
            if (ROOT/n).exists():assert digest(ROOT/n)==sha,n
            else:assert n.startswith('_stage2_') and n.endswith('_manifest.json'),n
    assert hashlib.sha256(raw).hexdigest()==plan['common']['baselines_source_manifest_sha256']
    assert plan['experiment']==EXPERIMENT and plan['preset']=='baseline2_repair' and plan['common']['baselines_run']=='baseline2' and plan['split']=='test' and not plan['smoke'] and plan['backend']=='hf'
    assert plan['thresholds']==CONVENTIONAL_THRESHOLDS and plan['repair_prompt']==REPAIR_VERSIONS['baseline2_repair'] and plan['generator_prompt']==generator_prompt_info(False)
    assert plan['limits']==Limits().to_dict() and not plan['adapter_hashes'] and set(plan['resolved_models'])=={'generator','agent'}
    for role,r in plan['resolved_models'].items():
        assert r['model']==MODELS[role]['id'] and r['revision']==MODELS[role]['revision']
        assert r['precision']==dict(quantization='none',compute_dtype='torch.float16',floating_parameter_dtypes=['torch.float16'])
    assert b['completion.json']['complete'] and b['completion.json']['tasks']==100
    ids=[t['id'] for _,t in tasks];assert plan['task_ids']==ids
    expected={str(p.relative_to(ROOT)):digest(p) for p,t in tasks};expected.update({t['data_file']:digest(ROOT/t['data_file']) for p,t in tasks});assert plan['inputs']==expected
    assert len([n for n in b if n.startswith('tasks/')])==len([n for n in b if n.startswith('generation/')])==100
    assert not [n for n in b if n.startswith('contracts/')]
    for n,sha in plan['runtime'].items():assert digest(ROOT/n)==sha,n
    bookpath=ROOT/'stage2-baseline2.ipynb'
    if not bookpath.exists():bookpath=ROOT/'notebooks/executed/stage2/baselines_hints_v1/stage2_baselines_hints_baseline2_kaggle_executed.ipynb'
    book=nbformat.read(bookpath,as_version=4);template=nbformat.read(ROOT/'notebooks/kaggle/stage2_baselines_hints/stage2_baselines_hints_baseline2_kaggle.ipynb',as_version=4)
    audit=dict(experiment=EXPERIMENT,zip_sha256=digest(path),source_manifest_sha256=hashlib.sha256(raw).hexdigest(),notebook=dict(source_matches=[c.source for c in book.cells]==[c.source for c in template.cells],errors=[dict(cell=i,error=o.get('evalue')) for i,c in enumerate(book.cells) for o in c.get('outputs',[]) if o.output_type=='error']),mismatches=[],cases={},comparisons={},limitations=['CPU Python 3.12 versus recorded Kaggle Python 3.13.16; replay validates observed cases.','Current comparisons change checker, parser, generator conditioning and contract-dependent repair features; not a single-component ablation.','Previously inspected tasks informed repair features; diagnostic evidence only.'])
    data={t['data_file']:pd.read_csv(ROOT/t['data_file']) for _,t in tasks}
    OUT.mkdir(parents=True,exist_ok=True)
    for index,(_,task) in enumerate(tasks,1):
        tid=task['id'];before=data[task['data_file']];item=b['tasks/'+tid+'.json'];row=item['record'];trace=item['trace'];g=b['generation/'+tid+'.json']
        assert row['system']=='baseline2_repair' and row['checker']=='conventional' and not row['generator_contract'] and row['schema_guard'] is None and row['repair_arm']=='evidence-no-contract'
        assert row['repair_version']==REPAIR_VERSIONS['baseline2_repair'] and row['parser_tokens']==row['parser_calls']==0 and row['tokens']==row['generation_repair_tokens']
        assert row['generation_repair_tokens']==g['prompt_tokens']+g['output_tokens']+sum(e['tokens'] for e in trace['events'] if e['event']=='llm')
        expected_messages=build_generator_messages({k:task[k] for k in ['request','data_file']},{task['data_file']:schema_entry(before)},None)
        if g['messages']!=expected_messages or g['raw_output']!=trace['generator_reply']:audit['mismatches'].append([tid,'generation_input'])
        ref=safe_run(task['reference_code'],before,**Limits().sandbox());assert ref.ran_ok,(tid,ref.error)
        checker=conventional_checker(before);reference_verdict=dataclasses.asdict(checker(ref.df_out));cache={}
        def execute(code):
            if code not in cache:
                result=safe_run(code,before,**Limits().sandbox());correct,reason=outputs_equal(ref.df_out,result.df_out) if result.ran_ok else (False,None)
                cache[code]=dict(ran_ok=result.ran_ok,error=result.error,correct=correct,equivalence=reason,verdict=dataclasses.asdict(checker(result.df_out)) if result.ran_ok else None)
            return cache[code]
        initial=execute(trace['initial_code']);final=execute(trace['final_code']) if trace.get('final_code') else None
        if initial['correct']!=row['initial_correct']:audit['mismatches'].append([tid,'initial_correct'])
        if row['outcome']=='ACCEPT' and (not final or final['correct']!=row['final_correct'] or (final['verdict'] or {}).get('status')!='PASS'):audit['mismatches'].append([tid,'accepted_final'])
        audit['cases'][tid]=dict(request=task['request'],record=row,reference_verdict=reference_verdict,initial=initial,final=final)
        if index%10==0:write(OUT/'audit.partial.json',audit);print('CPU audited',index,'/100',flush=True)
    rows={i:c['record'] for i,c in audit['cases'].items()}
    audit['summary']=dict(outcomes=dict(collections.Counter(r['outcome'] for r in rows.values())),initial_correct=sum(r['initial_correct'] for r in rows.values()),lost_initial_correct=[i for i,r in rows.items() if r['initial_correct'] and not r['final_correct']],repaired=[i for i,r in rows.items() if not r['initial_correct'] and r['final_correct']],wrong_accepts=[i for i,r in rows.items() if r['outcome']=='ACCEPT' and not r['final_correct']],reference_verdicts=dict(collections.Counter(c['reference_verdict']['status'] for c in audit['cases'].values())),families={f:dict(tasks=sum(r['transformation_type']==f for r in rows.values()),correct=sum(r['final_correct'] for r in rows.values() if r['transformation_type']==f),incorrect_accepted=sum(r['outcome']=='ACCEPT' and not r['final_correct'] for r in rows.values() if r['transformation_type']==f)) for f in sorted({r['transformation_type'] for r in rows.values()})})
    for method in ['gold','selective','tools']:
        o=load(ROOT/f'kaggle_outputs/stage2/repair_hints_v1/stage2_repair_hints_{method}.zip');op=o['plan.json'];old={v['record']['task_id']:v['record'] for n,v in o.items() if n.startswith('tasks/')}
        checks={k:plan[k]==op[k] for k in ['inputs','task_ids','environment','limits']};checks['generator']=plan['models']['generator']==op['models']['generator'];checks['agent']=plan['models']['agent']==op['models']['agent'];assert all(checks.values())
        audit['comparisons'][method]=dict(settings_match=checks,correct=sum(r['final_correct'] for r in old.values()),incorrect_accepted=sum(r['outcome']=='ACCEPT' and not r['final_correct'] for r in old.values()),withheld=sum(r['outcome']!='ACCEPT' for r in old.values()),tokens=sum(r['tokens'] for r in old.values()),only_baseline2_correct=[i for i in rows if rows[i]['final_correct'] and not old[i]['final_correct']],only_other_correct=[i for i in rows if old[i]['final_correct'] and not rows[i]['final_correct']])
    old_traces={r['task_id']:r for r in map(json.loads,(ROOT/'results/stage2_clean_v1/baseline1/traces.jsonl').read_text().splitlines())}
    audit['historical_direct_initial_agreement']=dict(same_initial_code=sum(b['tasks/'+i+'.json']['trace']['initial_code']==old_traces[i]['initial_code'] for i in rows),tasks=100,note='Historical direct generation is not automatically a matched current-stack baseline.')
    write(OUT/'audit.json',audit);print('Mismatches',audit['mismatches'],flush=True);assert not audit['mismatches']
if __name__=='__main__':main()
