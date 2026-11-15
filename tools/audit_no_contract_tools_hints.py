"""CPU replay and matched generator-contract ablation audit; no new model calls."""
import argparse,collections,dataclasses,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import nbformat
import pandas as pd
from tools.audit_baseline2_hints import load
from eval.stage2_selective.runner import load_tasks
from eval.stage2_combined_3b.adapters import SPECS
from eval.stage2_combined_3b.config import MODELS
from eval.stage2_baselines_hints.config import EXPERIMENT,REPAIR_VERSIONS
from eval.stage2_dual_3b.artifacts import write,digest
from eval.stage2_repair_dev.guard import guarded_verifier
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from igt.verifier import verify
from igt.agents.code_generator_v2 import build_generator_messages,generator_prompt_info
from finetune.selective_facts_scopefix.context import schema_entry
from eval.equivalence import outputs_equal
OUT=ROOT/'results/baselines_hints_v1/no_contract_tools_audit'

def strip_metadata(x):
    if isinstance(x,dict):return {k:strip_metadata(v) for k,v in x.items() if k not in ['model','seconds']}
    if isinstance(x,list):return [strip_metadata(v) for v in x]
    return x

def main(method='tools'):
    if method not in ['selective','tools']:raise ValueError('Unknown parser method')
    OUT=ROOT/f'results/baselines_hints_v1/no_contract_{method}_audit'
    name=f'stage2_baselines_hints_no_contract_{method}.zip';path=ROOT/name
    if not path.exists():path=ROOT/'kaggle_outputs/stage2/baselines_hints_v1'/name
    b=load(path);o=load(ROOT/f'kaggle_outputs/stage2/repair_hints_v1/stage2_repair_hints_{method}.zip');p=b['plan.json'];op=o['plan.json'];tasks=load_tasks('test');ids=[t['id'] for _,t in tasks]
    with zipfile.ZipFile(ROOT/'notebooks/kaggle/stage2_baselines_hints/stage2_baselines_hints_source.zip') as z:
        manifest_name=next(n for n in z.namelist() if n.endswith('/manifest.json') and n.count('/')==1);raw=z.read(manifest_name);prefix=manifest_name[:-len('manifest.json')]
        for n,sha in json.loads(raw)['sha256'].items():
            assert hashlib.sha256(z.read(prefix+n)).hexdigest()==sha
            if (ROOT/n).exists():assert digest(ROOT/n)==sha,n
            else:assert n.startswith('_stage2_') and n.endswith('_manifest.json'),n
    assert p['common']['baselines_source_manifest_sha256']==hashlib.sha256(raw).hexdigest()
    assert p['experiment']==EXPERIMENT and p['preset']=='no_contract_to_generator' and p['common']['baselines_run']=='no_contract_'+method and p['method']==method and p['arm']=='checked' and p['split']=='test' and not p['smoke'] and p['backend']=='hf'
    checks={k:p[k]==op[k] for k in ['models','adapter_hashes','inputs','task_ids','environment','limits','thresholds','repair_prompt','parser_settings']};assert all(checks.values())
    assert p['task_ids']==ids and p['adapter_hashes']==SPECS[method]['sha256']
    expected={str(f.relative_to(ROOT)):digest(f) for f,t in tasks};expected.update({t['data_file']:digest(ROOT/t['data_file']) for f,t in tasks});assert p['inputs']==expected
    for n,sha in p['runtime'].items():assert digest(ROOT/n)==sha,n
    for role in ['generator','agent']:
        r=p['resolved_models'][role];assert r['model']==MODELS[role]['id'] and r['revision']==MODELS[role]['revision'] and r['precision']==op['resolved_models'][role]['precision']
    assert p['resolved_models']['parser']['revision']==MODELS['parser']['revision'] and p['resolved_models']['parser']['precision']==op['resolved_models']['parser']['precision']
    assert p['generator_prompt']==generator_prompt_info(False) and p['repair_prompt']==REPAIR_VERSIONS['no_contract_to_generator']
    assert b['completion.json']['complete'] and b['completion.json']['tasks']==100
    assert len([n for n in b if n.startswith('tasks/')])==len([n for n in b if n.startswith('contracts/')])==100
    for tid in ids:assert strip_metadata(b['contracts/'+tid+'.json'])==strip_metadata(o['contracts/'+tid+'.json']),tid
    bookpath=ROOT/('stage-2-no-contract-tool-use.ipynb' if method=='tools' else 'stage2-selective-baseline.ipynb')
    if not bookpath.exists():bookpath=ROOT/f'notebooks/executed/stage2/baselines_hints_v1/stage2_baselines_hints_no_contract_{method}_kaggle_executed.ipynb'
    book=nbformat.read(bookpath,as_version=4);template=nbformat.read(ROOT/f'notebooks/kaggle/stage2_baselines_hints/stage2_baselines_hints_no_contract_{method}_kaggle.ipynb',as_version=4)
    audit=dict(experiment=EXPERIMENT,zip_sha256=digest(path),source_manifest_sha256=hashlib.sha256(raw).hexdigest(),matched_settings=checks,contracts_match_control=100,notebook=dict(source_matches=[c.source for c in book.cells]==[c.source for c in template.cells],errors=[dict(cell=i,error=x.get('evalue')) for i,c in enumerate(book.cells) for x in c.get('outputs',[]) if x.output_type=='error']),mismatches=[],cases={},limitations=['CPU Python 3.12 versus recorded Kaggle Python 3.13.16; results validate these observed cases only.','Previously inspected tasks informed the implementation; diagnostic evidence, not independent confirmation.'])
    OUT.mkdir(parents=True,exist_ok=True);data={t['data_file']:pd.read_csv(ROOT/t['data_file']) for _,t in tasks}
    for index,(_,task) in enumerate(tasks,1):
        tid=task['id'];before=data[task['data_file']];item=b['tasks/'+tid+'.json'];r=item['record'];trace=item['trace'];c=b['contracts/'+tid+'.json'];contract=c['extracted']['contract'] if c['extracted']['status']=='ok' else None
        assert r['system']=='no_contract_to_generator' and r['checker']=='intent' and r['generator_contract'] is False and r['schema_guard']=='declared-output-columns-v1' and r['repair_version']==REPAIR_VERSIONS['no_contract_to_generator']
        assert r['parser_tokens']==c['tokens'] and r['parser_calls']==c['llm_calls'] and r['tokens']==r['parser_tokens']+r['generation_repair_tokens']
        reference=safe_run(task['reference_code'],before,**Limits().sandbox());assert reference.ran_ok,(tid,reference.error)
        verdict=dataclasses.asdict(verify(contract,before,reference.df_out)[0]) if contract else None
        if verdict!=r['parser_reference_verdict']:audit['mismatches'].append([tid,'reference_verdict'])
        detail=dict(record=r,initial=None,final=None,reference_verdict=verdict)
        if trace:
            g=b['generation/'+tid+'.json'];expected=build_generator_messages({k:task[k] for k in ['request','data_file']},{task['data_file']:schema_entry(before)},None)
            if g['messages']!=expected or g['raw_output']!=trace['generator_reply']:audit['mismatches'].append([tid,'generation_input'])
            assert r['generation_repair_tokens']==g['prompt_tokens']+g['output_tokens']+sum(e['tokens'] for e in trace['events'] if e['event']=='llm')
            checker=guarded_verifier(lambda after:verify(contract,before,after)[0],contract,before)
            if dataclasses.asdict(checker(reference.df_out))!=r['combined_reference_verdict']:audit['mismatches'].append([tid,'guard_reference'])
            cache={}
            def execute(code):
                if code not in cache:
                    result=safe_run(code,before,**Limits().sandbox());correct,reason=outputs_equal(reference.df_out,result.df_out) if result.ran_ok else (False,None)
                    cache[code]=dict(ran_ok=result.ran_ok,error=result.error,correct=correct,equivalence=reason,verdict=dataclasses.asdict(checker(result.df_out)) if result.ran_ok else None)
                return cache[code]
            detail['initial']=execute(trace['initial_code'])
            if detail['initial']['correct']!=r['initial_correct']:audit['mismatches'].append([tid,'initial_correct'])
            if r['outcome']=='ACCEPT':
                detail['final']=execute(trace['final_code'])
                if detail['final']['correct']!=r['final_correct'] or (detail['final']['verdict'] or {}).get('status')!='PASS':audit['mismatches'].append([tid,'accepted_final'])
        else:assert r['stage_ended']=='parse' and r['initial_correct'] is None and not r['final_correct']
        audit['cases'][tid]=detail
        if index%10==0:write(OUT/'audit.partial.json',audit);print('Audited',index,'/100',flush=True)
    rows={tid:c['record'] for tid,c in audit['cases'].items()};old={tid:o['tasks/'+tid+'.json']['record'] for tid in ids}
    audit['summary']=dict(initial_correct=sum(r['initial_correct'] is True for r in rows.values()),lost_initial_correct=[i for i,r in rows.items() if r['initial_correct'] and not r['final_correct']],repaired=[i for i,r in rows.items() if r['initial_correct'] is False and r['final_correct']],wrong_accepts=[i for i,r in rows.items() if r['outcome']=='ACCEPT' and not r['final_correct']],outcomes=dict(collections.Counter(r['outcome'] for r in rows.values())),only_no_contract_correct=[i for i in ids if rows[i]['final_correct'] and not old[i]['final_correct']],only_proposed_correct=[i for i in ids if old[i]['final_correct'] and not rows[i]['final_correct']])
    write(OUT/'audit.json',audit);print('Mismatches',audit['mismatches'],flush=True);assert not audit['mismatches']
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['selective','tools'],default='tools')
    main(parser.parse_args().method)
