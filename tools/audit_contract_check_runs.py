"""Validate, replay contract correction policy, and CPU-score uploaded checked runs."""
import collections,copy,dataclasses,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import nbformat
import pandas as pd
from eval.stage2_contract_check.config import EXPERIMENT,COMMON,MODELS,PARSER_SETTINGS,PROTOCOLS,REPAIR_VERSION,THRESHOLDS
from eval.stage2_contract_check.artifacts import runtime_hashes
from eval.stage2_contract_check.correction import check_and_correct
from eval.stage2_contract_check.diagnostics import diagnose
from eval.stage2_combined_3b.adapters import SPECS
from eval.stage2_combined_3b.collect import summarize
from eval.stage2_selective.runner import load_tasks
from eval.stage2_dual_3b.artifacts import digest,write
from eval.stage2_repair_dev.guard import guarded_verifier
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from igt.verifier import verify
from igt.agents.code_generator_v2 import build_generator_messages,generator_prompt_info
from finetune.selective_facts_scopefix.context import schema_entry
from eval.equivalence import outputs_equal
OUT=ROOT/'results/contract_check_v1/completed'

def locate(method):
    name='stage2_contract_check_'+method+'_checked.zip'
    path=ROOT/name
    return path if path.exists() else ROOT/'kaggle_outputs/stage2/contract_check_v1'/name

def load(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        raw={n:z.read(n) for n in z.namelist()}
    seal=json.loads(raw.pop('bundle.json'));assert seal['complete'] and seal['sha256']=={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}
    assert 'failure.json' not in raw
    return {n:json.loads(b) for n,b in raw.items() if n.endswith('.json')}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ROOT/'notebooks/kaggle/stage2_contract_check/stage2_contract_check_source.zip') as z:
        name=next(n for n in z.namelist() if n.endswith('/manifest.json') and n.count('/')==1);manifest=z.read(name);prefix=name[:-len('manifest.json')]
        for n,sha in json.loads(manifest)['sha256'].items():
            assert hashlib.sha256(z.read(prefix+n)).hexdigest()==sha,n
            if (ROOT/n).exists():assert digest(ROOT/n)==sha,n
            else:assert n=='_stage2_combined_3b_manifest.json',n
    source_sha=hashlib.sha256(manifest).hexdigest()
    tasks=load_tasks('test');data={t['data_file']:pd.read_csv(ROOT/t['data_file']) for _,t in tasks}
    audit=dict(experiment=EXPERIMENT,source_manifest_sha256=source_sha,zip_sha256={},notebooks={},methods={},mismatches=[],cases={},limitations=['Previously inspected tasks informed these rules; results are development diagnostics, not independent confirmation.','CPU Python 3.12 versus Kaggle 3.13.16; checks establish observed-case agreement only.','Correction selection and issue labels are policies/heuristics, not independent guarantees of request fidelity.'])
    bundles={};controls={}
    for method in ['selective','tools']:
        path=locate(method);b=load(path);bundles[method]=b
        o=load(ROOT/f'kaggle_outputs/stage2/combined_3b_v1/stage2_combined_3b_{method}_proposed.zip');controls[method]=o;p=b['plan.json'];op=o['plan.json']
        assert p['experiment']==EXPERIMENT and p['method']==method and p['arm']=='checked' and p['preset']=='proposed' and not p['smoke'] and p['split']=='test' and p['backend']=='hf'
        assert p['source_manifest_sha256']==source_sha and p['runtime']==runtime_hashes()
        assert p['common']==COMMON and p['models']==MODELS and p['parser_settings']==PARSER_SETTINGS[method] and p['protocol']==PROTOCOLS[method]
        assert p['repair_prompt']==REPAIR_VERSION and p['generator_prompt']==generator_prompt_info(True) and p['limits']==Limits().to_dict() and p['thresholds']==THRESHOLDS
        assert p['adapter_hashes']==SPECS[method]['sha256']
        checks={k:p[k]==op[k] for k in ['inputs','task_ids','models','adapter_hashes','environment','generator_prompt','repair_prompt','limits','thresholds']};assert all(checks.values())
        ids=[t['id'] for _,t in tasks];assert p['task_ids']==ids
        expected={str(path.relative_to(ROOT)):digest(path) for path,t in tasks};expected.update({t['data_file']:digest(ROOT/t['data_file']) for _,t in tasks});assert p['inputs']==expected
        assert b['completion.json']['complete'] and b['completion.json']['tasks']==100
        for section in ['tasks','contracts']:assert len([n for n in b if n.startswith(section+'/')])==100
        for role in ['generator','agent']:
            r=p['resolved_models'][role];assert r['revision']==MODELS[role]['revision'] and r['model']==MODELS[role]['id']
            assert r['precision']==op['resolved_models'][role]['precision']
        assert p['resolved_models']['parser']['revision']==MODELS['parser']['revision']
        bookpath=ROOT/('stage2-check-selective.ipynb' if method=='selective' else 'stage2-tool-contract-check.ipynb')
        if not bookpath.exists():bookpath=ROOT/f'notebooks/executed/stage2/contract_check_v1/stage2_contract_check_{method}_kaggle_executed.ipynb'
        book=nbformat.read(bookpath,as_version=4);template=nbformat.read(ROOT/f'notebooks/kaggle/stage2_contract_check/stage2_contract_check_{method}_kaggle.ipynb',as_version=4)
        audit['notebooks'][method]=dict(source_matches=[c.source for c in book.cells]==[c.source for c in template.cells],errors=[dict(cell=i,error=x.get('evalue')) for i,c in enumerate(book.cells) for x in c.get('outputs',[]) if x.output_type=='error'])
        rows=[v['record'] for n,v in b.items() if n.startswith('tasks/')];oldrows={v['record']['task_id']:v['record'] for n,v in o.items() if n.startswith('tasks/')};now={r['task_id']:r for r in rows}
        summary=summarize(rows);summary.update(initial_correct=sum(r['initial_correct'] is True for r in rows),decisions=dict(collections.Counter(r['check_decision'] for r in rows)),correction_calls=sum(r['check_correction_attempted'] for r in rows),correction_tokens=sum(r['check_tokens'] for r in rows),gains=[i for i in ids if now[i]['final_correct'] and not oldrows[i]['final_correct']],losses=[i for i in ids if oldrows[i]['final_correct'] and not now[i]['final_correct']],previous_correct_initial_now_parseblocked=[i for i in ids if oldrows[i]['initial_correct'] is True and now[i]['stage_ended']=='parse'],settings_match=checks,wrong_accepts=[r['task_id'] for r in rows if r['outcome']=='ACCEPT' and not r['final_correct']])
        summary['attempted_changed']=[];summary['attempted_unchanged']=[];summary['selected_changed']=[];summary['reference_compatibility_gains']=[]
        for tid in ids:
            c=b['contracts/'+tid+'.json'];ch=c['contract_check'];first=c.get('first_extracted',c['extracted']);oldc=o['contracts/'+tid+'.json'];assert first==oldc['extracted']
            if ch['attempted']:
                summary['attempted_unchanged' if first==ch['correction_extracted'] else 'attempted_changed'].append(tid)
                if ch['decision']=='use_corrected' and first!=ch['correction_extracted']:summary['selected_changed'].append(tid)
            r=now[tid]
            if (r.get('first_parser_reference_verdict') or {}).get('status') not in [None,'PASS'] and (r.get('parser_reference_verdict') or {}).get('status')=='PASS':summary['reference_compatibility_gains'].append(tid)
        audit['methods'][method]=summary;audit['zip_sha256'][method]=digest(path)
    for index,(_,task) in enumerate(tasks,1):
        tid=task['id'];before=data[task['data_file']];ref=safe_run(task['reference_code'],before,**Limits().sandbox());assert ref.ran_ok,(tid,ref.error)
        audit['cases'][tid]=dict(request=task['request'],methods={})
        for method in ['selective','tools']:
            b=bundles[method];c=b['contracts/'+tid+'.json'];item=b['tasks/'+tid+'.json'];r=item['record'];trace=item['trace'];ch=c['contract_check'];first=copy.deepcopy(c)
            first.pop('contract_check');first['extracted']=first.pop('first_extracted',first['extracted']);first['tokens']-=ch['tokens'];first['llm_calls']-=ch['llm_calls']
            public={k:task[k] for k in ['id','request','data_file']}
            class Reply:
                model='saved-correction-CPU-replay'
                def prompt_tokens(self,msg):return ch.get('needed_tokens',1024)-1024 if ch.get('context_limit') else 0
                def complete(self,*args,**kwargs):return ch['correction_raw'],ch['prompt_tokens'],ch['output_tokens']
            replay,check=check_and_correct(public,before,first,Reply(),PARSER_SETTINGS[method]['context_tokens'])
            for key in ['decision','attempted','first_issues','remaining_issues','correction_messages','correction_extracted','tokens','llm_calls']:
                if check.get(key)!=ch.get(key):audit['mismatches'].append([method,tid,'policy',key])
            if replay['extracted']!=c['extracted']:audit['mismatches'].append([method,tid,'selected_contract'])
            assert r['parser_tokens']==c['tokens'] and r['parser_calls']==c['llm_calls'] and r['check_tokens']==ch['tokens'] and r['tokens']==r['parser_tokens']+r['generation_repair_tokens']
            contract=c['extracted']['contract'] if c['extracted']['status']=='ok' else None
            v=dataclasses.asdict(verify(contract,before,ref.df_out)[0]) if contract else None
            if v!=r['parser_reference_verdict']:audit['mismatches'].append([method,tid,'reference_verdict'])
            detail=dict(record=r,check=ch,reference_verdict=v,initial=None,final=None)
            if trace:
                g=b['generation/'+tid+'.json'];expected=build_generator_messages(public,{task['data_file']:schema_entry(before)},contract)
                if g['messages']!=expected or g['raw_output']!=trace['generator_reply']:audit['mismatches'].append([method,tid,'generation_input'])
                assert r['generation_repair_tokens']==g['prompt_tokens']+g['output_tokens']+sum(e['tokens'] for e in trace['events'] if e['event']=='llm')
                checker=guarded_verifier(lambda after:verify(contract,before,after)[0],contract,before)
                if dataclasses.asdict(checker(ref.df_out))!=r['combined_reference_verdict']:audit['mismatches'].append([method,tid,'guard_reference'])
                cache={}
                def execute(code):
                    if code not in cache:
                        result=safe_run(code,before,**Limits().sandbox());correct,reason=outputs_equal(ref.df_out,result.df_out) if result.ran_ok else (False,None)
                        cache[code]=dict(ran_ok=result.ran_ok,error=result.error,correct=correct,equivalence=reason,rows=len(result.df_out) if result.ran_ok else None,verdict=dataclasses.asdict(checker(result.df_out)) if result.ran_ok else None)
                    return cache[code]
                detail['initial']=execute(trace['initial_code'])
                if detail['initial']['correct']!=r['initial_correct']:audit['mismatches'].append([method,tid,'initial_correct'])
                if r['outcome']=='ACCEPT':
                    detail['final']=execute(trace['final_code'])
                    if detail['final']['correct']!=r['final_correct'] or (detail['final']['verdict'] or {}).get('status')!='PASS':audit['mismatches'].append([method,tid,'accepted_final'])
            audit['cases'][tid]['methods'][method]=detail
        if index%10==0:write(OUT/'audit.partial.json',audit);print('Audited',index,'/100 pairs',flush=True)
    # Generalization probe: an object column holding actual integers can cast dtype without changing values.
    before=pd.DataFrame({'x':pd.Series([1,2],dtype=object),'label':['a','b']});after=before.copy();after['x']=after.x.astype(int)
    contract=dict(contract_version='1.0',must_change=[dict(type='column_dtype',target='x',dtype='int')],must_preserve=[dict(type='column_values_unchanged',target='x'),dict(type='row_count',op='unchanged'),dict(type='columns_unchanged',target='*',except_=['x'])])
    contract['must_preserve'][-1]['except']=contract['must_preserve'][-1].pop('except_')
    audit['diagnostic_false_positive_probe']=dict(request='Store x as integers, preserving its numerical values.',verdict=dataclasses.asdict(verify(contract,before,after)[0]),issues=diagnose('Store x as integers, preserving its numerical values.',dict(status='ok',contract=contract),before),note='A proof label is not universally sound: this correct cast passes the verifier. No submitted run is modified.')
    write(OUT/'audit.json',audit);print('Mismatches:',audit['mismatches'],flush=True);assert not audit['mismatches']
if __name__=='__main__':main()
