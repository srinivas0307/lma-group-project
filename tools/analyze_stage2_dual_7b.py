"""Strictly collect and CPU-audit both 7B NF4 repair Stage 2 runs without new model inference."""
import collections
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import nbformat
import pandas as pd
from eval.stage2_dual.artifacts import digest,write
from eval.stage2_dual.runner import load_tasks
from eval.stage2_dual.parsers import parse
from eval.stage2_dual.collect import summarize
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from igt.verifier import verify
from eval.equivalence import outputs_equal
from igt.agents.code_generator_v2 import build_generator_messages
from finetune.selective_facts_scopefix.context import schema_entry
OUT=ROOT/'results/stage2_dual_7b_analysis'


def input_diagnostics():
    from finetune.selective_facts_scopefix.context import selective_facts,FactUnavailable
    from eval.stage2_tools.data_tools import DataTools
    results={}
    for _,task in load_tasks():
        if task['id']=='taxi_miss_005':
            before=pd.read_csv(ROOT/task['data_file'])
            try:original=selective_facts(task['request'],before)
            except FactUnavailable as exc:original=dict(error=str(exc))
            rephrased=task['request'].replace("both mean 'not recorded'","both denote 'not recorded'")
            results['router_false_positive']=dict(task_id=task['id'],original_request=task['request'],original_result=original,rephrased_request=rephrased,rephrased_result=selective_facts(rephrased,before),note='CPU-only lexical diagnosis. No modified-prompt model run or replacement score.')
        if task['id']=='titanic_miss_005':
            observation=DataTools(pd.read_csv(ROOT/task['data_file'])).call('group_profile',dict(column='Age',by=['Pclass']))
            results['group_query']=dict(task_id=task['id'],tool='group_profile',args=dict(column='Age',by=['Pclass']),observation=observation,note='Audit-only query; this was not requested by the model in the submitted run.')
    write(OUT/'input_diagnostics.json',results)
    return results



def compare_repair_models():
    comparison={}
    for method in ['selective','tools']:
        bundles={}
        for size,path in [('7b',ROOT/f'kaggle_outputs/stage2/dual_parser_v1/stage2_dual_{method}_proposed.zip'),('3b',ROOT/f'kaggle_outputs/stage2/dual_parser_3b_fp16_v1/stage2_dual_3b_{method}_proposed.zip')]:
            with zipfile.ZipFile(path) as z:
                bundles[size]={n:json.loads(z.read(n)) for n in z.namelist() if n.endswith('.json')}
        new,old=bundles['7b'],bundles['3b']
        def rs(bundle):return {r['record']['task_id']:r for n,r in bundle.items() if n.startswith('tasks/')}
        nr,orr=rs(new),rs(old)
        checks={key:new['plan.json'][key]==old['plan.json'][key] for key in ['parser_settings','generator_prompt','repair_prompt','limits','thresholds','inputs','task_ids','split','smoke','adapter_hashes']}
        checks['same_parser_payload_except_label_and_time']=all({k:v for k,v in r.items() if k not in ['model','seconds']}=={k:v for k,v in old[n].items() if k not in ['model','seconds']} for n,r in new.items() if n.startswith('contracts/'))
        checks['same_generation_payload_except_time']=all({k:v for k,v in r.items() if k!='seconds'}=={k:v for k,v in old[n].items() if k!='seconds'} for n,r in new.items() if n.startswith('generation/'))
        checks['same_initial_code']=all((r['trace'] or {}).get('initial_code')==(orr[t]['trace'] or {}).get('initial_code') for t,r in nr.items())
        if not all(checks.values()):raise ValueError(checks)
        comparison[method]=dict(checks=checks,models_7b=new['plan.json']['models'],models_3b=old['plan.json']['models'],
            gains=[t for t,r in nr.items() if r['record']['final_correct'] and not orr[t]['record']['final_correct']],
            losses=[t for t,r in nr.items() if orr[t]['record']['final_correct'] and not r['record']['final_correct']],
            summaries={size:summarize([r['record'] for r in rs(bundle).values()]) for size,bundle in bundles.items()})
    write(OUT/'repair_model_comparison.json',comparison)
    return comparison


def main():
    paths={m:ROOT/f'kaggle_outputs/stage2/dual_parser_v1/stage2_dual_{m}_proposed.zip' for m in ['selective','tools']}
    source=ROOT/'notebooks/kaggle/stage2_dual/stage2_dual_source.zip'
    OUT.mkdir(parents=True,exist_ok=True)
    input_diagnostics()
    compare_repair_models()
    # Run the unmodified real-result collector in the exact sealed source package.
    with tempfile.TemporaryDirectory(prefix='stage2_7b_collect_') as temp:
        with zipfile.ZipFile(source) as z:
            z.extractall(temp);manifest=z.read('stage2_dual_source/manifest.json')
        isolated=Path(temp)/'stage2_dual_source';(isolated/'_stage2_dual_manifest.json').write_bytes(manifest)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(isolated),str(isolated/'src')]))
        destination=OUT/'comparison'
        if destination.exists():
            # Reproduce strict collection in a fresh temp folder; preserve/compare existing report contents.
            destination=Path(temp)/'comparison'
        result=subprocess.run([sys.executable,'-m','eval.stage2_dual.collect',*map(str,paths.values()),'--out',str(destination)],cwd=isolated,env=env,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        collector=json.loads((destination/'summary.json').read_text())
        if destination!=OUT/'comparison':
            if json.loads((OUT/'comparison/summary.json').read_text())!=collector:raise ValueError('Existing collector output differs')
        print(result.stdout,flush=True)
    payload={};plans={};records={};contracts={};generations={}
    for method,path in paths.items():
        with zipfile.ZipFile(path) as z:
            payload[method]={n:json.loads(z.read(n)) for n in z.namelist() if n.endswith('.json') and n!='bundle.json'}
        plans[method]=payload[method]['plan.json']
        for name,sha in plans[method]['runtime'].items():
            if digest(ROOT/name)!=sha:raise ValueError('Different local runtime: '+name)
        records[method]={r['record']['task_id']:r for n,r in payload[method].items() if n.startswith('tasks/')}
        contracts[method]={r['task_id']:r for n,r in payload[method].items() if n.startswith('contracts/')}
        generations[method]={r['task_id']:r for n,r in payload[method].items() if n.startswith('generation/')}
    audit=dict(collector=collector,zip_sha256={m:digest(p) for m,p in paths.items()},notebooks={},policy_mismatches=[],score_mismatches=[],generation_mismatches=[],cases={},summary={},families={},
       limits=['CPU replay uses local Python/pandas; exact mismatches are retained.','Token-budget stopping is replayed at recorded locations because tokenizer per-call counts are absent.','Tool top-value display ordering/cutoffs can vary among tied frequencies; statistical fields are compared independently.','These test tasks have repeatedly informed development; no independent research claim.'])
    tasks=load_tasks();data={t['data_file']:pd.read_csv(ROOT/t['data_file']) for _,t in tasks}
    for method in paths:
        executed=ROOT/f'notebooks/executed/stage2/dual_parser_v1/stage2_dual_{method}_proposed_kaggle_executed.ipynb'
        book=nbformat.read(executed,as_version=4);template=nbformat.read(ROOT/f'notebooks/kaggle/stage2_dual/stage2_dual_{method}_proposed_kaggle.ipynb',as_version=4)
        audit['notebooks'][method]=dict(source_matches=[c.source for c in book.cells]==[c.source for c in template.cells],errors=[dict(cell=i,error=o.get('evalue')) for i,c in enumerate(book.cells) for o in c.get('outputs',[]) if o.output_type=='error'])
        group=[r['record'] for r in records[method].values()];summary=summarize(group)
        by_status=collections.Counter(r['extracted']['status'] for r in contracts[method].values())
        summary.update(parser_statuses=dict(by_status),reference_verdicts=dict(collections.Counter((r.get('parser_reference_verdict') or {}).get('status','no_contract') for r in group)),
           initially_correct=sum(r['initial_correct'] is True for r in group),repair_agent_calls=sum(r['llm_calls']-r['parser_calls']-(0 if r['stage_ended']=='parse' else 1) for r in group),
           repair_agent_tools=sum(r.get('tool_calls',0) for r in group),
           withheld_due_parse=sum(r['stage_ended']=='parse' for r in group),withheld_reference_incompatible=sum(r['outcome']!='ACCEPT' and r.get('parser_reference_verdict') is not None and r['parser_reference_verdict']['status']!='PASS' for r in group),
           withheld_reference_compatible=sum(r['outcome']!='ACCEPT' and (r.get('parser_reference_verdict') or {}).get('status')=='PASS' for r in group),
           wrong_accepts=[r['task_id'] for r in group if r['outcome']=='ACCEPT' and not r['final_correct']],
           parser_tool_tasks=[tid for tid,c in contracts[method].items() if c.get('tool_calls')],placeholder_calls=sum(t['args']=={'column':'__unknown_column__'} for c in contracts[method].values() for t in c.get('tool_calls',[])))
        audit['summary'][method]=summary
    cache_dir=Path(tempfile.gettempdir())/('igt_stage2_7b_'+hashlib.sha256(json.dumps(audit['zip_sha256'],sort_keys=True).encode()).hexdigest()[:12]);cache_dir.mkdir(exist_ok=True)
    for index,(path,task) in enumerate(tasks,1):
        tid=task['id'];before=data[task['data_file']]
        reference=safe_run(task['reference_code'],before,**Limits().sandbox())
        if not reference.ran_ok:raise ValueError('Reference failed: '+tid+str(reference.error))
        case=dict(request=task['request'],family=task['transformation_type'],reference_rows=len(reference.df_out),reference_columns=list(reference.df_out.columns),methods={})
        code_results={}
        def execute(code):
            key=hashlib.sha256((tid+'\0'+code+'\0'+sys.version+'\0'+pd.__version__).encode()).hexdigest();cached=cache_dir/(key+'.json')
            if code not in code_results:
                if cached.exists():item=json.loads(cached.read_text())
                else:
                    result=safe_run(code,before,**Limits().sandbox())
                    correct,reason=outputs_equal(reference.df_out,result.df_out) if result.ran_ok else (False,None)
                    item=dict(ran_ok=result.ran_ok,status=result.status,error=result.error,correct=correct,equivalence=reason,rows=len(result.df_out) if result.ran_ok else None,
                        columns=list(result.df_out.columns) if result.ran_ok else None,extra_columns=[c for c in result.df_out if c not in reference.df_out] if result.ran_ok else [])
                    # Gold and predicted-verifier verdicts matter for accepted wrong outputs.
                    if result.ran_ok:
                        item['gold_verdict']=dataclasses.asdict(verify(task['gold_contract'],before,result.df_out)[0])
                        item['predicted_verdicts']={m:dataclasses.asdict(verify(contracts[m][tid]['extracted']['contract'],before,result.df_out)[0]) if contracts[m][tid]['extracted']['status']=='ok' else None for m in paths}
                    write(cached,item)
                code_results[code]=item
            return code_results[code]
        for method in paths:
            saved=contracts[method][tid];r=records[method][tid]['record'];trace=records[method][tid]['trace']
            if method=='selective':replies=[saved['raw_output']] if saved['llm_calls'] else []
            else:replies=[m['content'] for m in saved['messages'] if m['role']=='assistant']
            class Replay:
                model='CPU-saved-reply-replay'
                def __init__(self):self.i=0
                def prompt_tokens(self,msg):return 24000 if self.i==len(replies) and saved['extracted']['status']=='parser_token_limit' else 0
                def complete(self,*args,**kw):text=replies[self.i];self.i+=1;return text,0,0
            replay=parse(method,task,before,Replay())
            for key in ['extracted','normalizations','llm_calls']:
                if replay.get(key)!=saved.get(key):audit['policy_mismatches'].append([method,tid,key])
            if method=='selective' and replay.get('messages')!=saved.get('messages'):audit['policy_mismatches'].append([method,tid,'messages'])
            if method=='tools':
                if replay['messages'][:2]!=saved['messages'][:2] or [(t['tool'],t['args']) for t in replay['tool_calls']]!=[(t['tool'],t['args']) for t in saved['tool_calls']]:audit['policy_mismatches'].append([method,tid,'tool_actions'])
                def strip_top(x):
                    if isinstance(x,dict):return {k:strip_top(v) for k,v in x.items() if k!='top_values'}
                    if isinstance(x,list):return [strip_top(v) for v in x]
                    return x
                if strip_top(replay['tool_calls'])!=strip_top(saved['tool_calls']):audit['policy_mismatches'].append([method,tid,'tool_observations'])
            contract=saved['extracted']['contract'] if saved['extracted']['status']=='ok' else None
            verdict=dataclasses.asdict(verify(contract,before,reference.df_out)[0]) if contract else None
            if verdict!=r['parser_reference_verdict']:audit['score_mismatches'].append([method,tid,'reference_verdict'])
            detail=dict(record=r,contract=contract,reference_verdict=verdict,initial=None,accepted_final=None)
            if trace is not None:
                expected=build_generator_messages(dict(request=task['request'],data_file=task['data_file']),{task['data_file']:schema_entry(before)},contract)
                if generations[method][tid]['messages']!=expected or generations[method][tid]['raw_output']!=trace['generator_reply']:audit['generation_mismatches'].append([method,tid])
                initial=execute(trace['initial_code']);detail['initial']=initial
                if initial['correct']!=r['initial_correct']:audit['score_mismatches'].append([method,tid,'initial_correct'])
                if r['outcome']=='ACCEPT':
                    final=execute(trace['final_code']);detail['accepted_final']=final
                    if final['correct']!=r['final_correct']:audit['score_mismatches'].append([method,tid,'final_correct'])
                    if (final.get('predicted_verdicts',{}).get(method) or {}).get('status')!='PASS':audit['score_mismatches'].append([method,tid,'accepted_verifier'])
            case['methods'][method]=detail
        audit['cases'][tid]=case
        if index%10==0:print('CPU reference/code replay',index,'/100 for both methods',flush=True)
    for method in paths:
        for family in sorted({c['family'] for c in audit['cases'].values()}):
            group=[c['methods'][method]['record'] for c in audit['cases'].values() if c['family']==family]
            audit['families'].setdefault(family,{})[method]=summarize(group)
    audit['paired']=dict(only_selective_correct=[tid for tid,c in audit['cases'].items() if c['methods']['selective']['record']['final_correct'] and not c['methods']['tools']['record']['final_correct']],
       only_tools_correct=[tid for tid,c in audit['cases'].items() if c['methods']['tools']['record']['final_correct'] and not c['methods']['selective']['record']['final_correct']],
       differing_contracts=[tid for tid in contracts['selective'] if contracts['selective'][tid]['extracted'].get('contract')!=contracts['tools'][tid]['extracted'].get('contract')])
    write(OUT/'audit.json',audit)
    print('SAVED',OUT)
    print('CHECKS',{k:len(audit[k]) for k in ['policy_mismatches','score_mismatches','generation_mismatches']})
    print('FAMILIES',json.dumps(audit['families']))
if __name__=='__main__':main()
