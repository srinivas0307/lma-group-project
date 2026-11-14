"""Audit tool-use training, bind its identical base to the prior audit, CPU-score both checkpoints."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
import nbformat
import pandas as pd
from analyze_selective_facts_scopefix import sealed
from finetune.tool_use.adapter import validate
from finetune.tool_use.config import MODELS
from finetune.tool_use.parser import parse_fresh
from finetune.tool_use.evaluate import development_tasks,summarize,eligible
from finetune.selective_facts.provenance import digest,write
from eval.contract_quality.harness import prepare,probe
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from igt.verifier import verify
from igt.agents.intent_parser import extract_contract

OUT=ROOT/'results/tool_use_v2_analysis/train_1_5b'

def main():
    source=ROOT/'kaggle_outputs/tool_use_v2/tool_use_train_1_5b.zip';files=sealed(source)
    import zipfile
    with zipfile.ZipFile(source) as z:
        selection=files['selection.json'];info=files['run_info.json']
        if info['size']!='1_5b' or info['smoke'] or not files['completion.json']['complete']:raise ValueError('Wrong run')
        with tempfile.TemporaryDirectory(prefix='tool_adapter_audit_') as tmp:
            folder=Path(tmp)
            for n in z.namelist():
                if n.startswith('adapter/'):
                    relative=Path(n).relative_to('adapter')
                    if len(relative.parts)!=1:raise ValueError('Invalid adapter member')
                    (folder/relative).write_bytes(z.read(n))
            validate(folder,'1_5b')
            for n,h in selection['selected']['adapter_sha256'].items():
                if digest(folder/n)!=h:raise ValueError('Selected candidate weight/protocol hash mismatch')
        logs=[json.loads(l) for l in z.read('train_log.jsonl').decode().splitlines()]
        for r in logs:
            for k in ['loss','eval_loss','grad_norm']:
                if k in r and not math.isfinite(r[k]):raise ValueError('Nonfinite training log')
    for n,h in info['source_hashes'].items():
        if digest(ROOT/n)!=h:raise ValueError('Changed training source: '+n)
    data=ROOT/'finetune/data/tool_use_v2'
    if digest(data/'manifest.json')!=info['data_manifest_sha256']:raise ValueError('Changed data manifest')
    for n,h in info['data_manifest']['sha256'].items():
        if digest(data/n)!=h:raise ValueError('Changed SFT data: '+n)
    book=nbformat.read(ROOT/'notebooks/executed/tool_use_v2/tool_use_train_1_5b_executed.ipynb',as_version=4)
    template=nbformat.read(ROOT/'notebooks/kaggle/tool_use_v2/tool_use_train.ipynb',as_version=4)
    original=sealed(ROOT/'kaggle_outputs/tool_use_v2/tool_use_base_1_5b.zip')
    baseline={r['task_id']:r for n,r in files.items() if n.startswith('baseline/tasks/')}
    previous={r['task_id']:r for n,r in original.items() if n.startswith('tasks/')}
    prior_audit_path=ROOT/'results/tool_use_v2_analysis/base_comparison/audit.json';prior_audit=json.loads(prior_audit_path.read_text())
    if prior_audit['score_mismatches'] or prior_audit['counter_mismatches']:raise ValueError('Prior base audit not clean')
    for tid,r in baseline.items():
        prev=previous[tid]
        for k in ['all_correct_accepted','any_wrong_accepted','parse_status','predicted_contract','request']:
            if r[k]!=prev[k]:raise ValueError('Different base control')
        if r['parser']['messages'][:2]!=prev['parser']['messages'][:2] or r['parser']['raw_output']!=prev['parser']['raw_output']:raise ValueError('Base prediction differs')
    stages=['semantic/'+c['name'] for c in selection['all_candidates']]
    rows={stage:{r['task_id']:r for n,r in files.items() if n.startswith(stage+'/tasks/')} for stage in stages}
    audit=dict(bundle_sha256=digest(source),complete=files['completion.json'],selection=selection,run_info={k:v for k,v in info.items() if k not in ['source_hashes','data_manifest','training_arguments']},
       notebook_source_matches=[c.source for c in book.cells]==[c.source for c in template.cells],notebook_errors=[dict(cell=i,error=o.get('evalue')) for i,c in enumerate(book.cells) for o in c.get('outputs',[]) if o.output_type=='error'],
       prior_base_audit_sha256=digest(prior_audit_path),base_predictions_identical=True,summary_mismatches=[],score_mismatches=[],control_mismatches=[],cases={},stats={})
    all_stages={'baseline':baseline,**rows};tasks=development_tasks();expected={t['id'] for t,_,_ in tasks}
    for stage,records in all_stages.items():
        if set(records)!=expected:raise ValueError('Coverage mismatch')
        summary=summarize(list(records.values()))
        for cohort,s in summary.items():
            group=[r for r in records.values() if r['cohort']==cohort]
            s.update(tasks_with_tools=sum(r['tool_attempts']>0 for r in group),tool_attempts=sum(r['tool_attempts'] for r in group),tool_failures=sum(r['tool_failures'] for r in group))
        if summary!=files[stage+'/summary.json']['summary']:audit['summary_mismatches'].append(stage)
        prov=files[stage+'/summary.json']['provenance']
        for group in ['inputs','runtime','original_task_hashes']:
            for n,h in prov[group].items():
                if digest(ROOT/n)!=h:raise ValueError('Changed evaluation source/input: '+n)
        if stage!='baseline':
            candidate=next(c for c in selection['all_candidates'] if stage.endswith(c['name']))
            if candidate['eligible']!=eligible(summary,selection['baseline']):raise ValueError('Eligibility differs')
            if candidate['summary']!=summary or candidate['adapter_sha256']!=prov['adapter_hashes']:raise ValueError('Checkpoint scoring provenance differs')
    for index,(task,cohort,root) in enumerate(tasks,1):
        tid=task['id'];before=pd.read_csv(root/task['data_file']);prepared=None;case=dict(request=task['request'],cohort=cohort,pattern=task.get('pattern',task.get('transformation_type')),stages={})
        if cohort=='original':
            ref=safe_run(task['reference_code'],before,**Limits().sandbox())
            if not ref.ran_ok:raise ValueError(ref.error)
        for stage in stages:
            r=rows[stage][tid];saved=r['parser'];replies=[m['content'] for m in saved['messages'] if m['role']=='assistant']
            class Replay:
                model='saved-replay'
                def __init__(self):self.i=0
                def prompt_tokens(self,m):return 0
                def complete(self,*args,**kw):text=replies[self.i];self.i+=1;return text,0,0
            replay=parse_fresh(task,before,Replay(),'p1-v2')
            for key in ['extracted','invalid_responses','llm_calls','stop_reason']:
                if replay[key]!=saved[key]:audit['control_mismatches'].append([stage,tid,key])
            if [(t['tool'],t['args']) for t in replay['tool_calls']]!=[(t['tool'],t['args']) for t in saved['tool_calls']]:audit['control_mismatches'].append([stage,tid,'tool_actions'])
            if replay['messages'][:2]!=saved['messages'][:2]:audit['control_mismatches'].append([stage,tid,'initial_messages'])
            # Check all observation fields except tied top-value display ordering/membership.
            for actual,recorded in zip(replay['tool_calls'],saved['tool_calls']):
                def strip_top(o):
                    if isinstance(o,dict):return {k:strip_top(v) for k,v in o.items() if k!='top_values'}
                    if isinstance(o,list):return [strip_top(x) for x in o]
                    return o
                if strip_top(actual['observation'])!=strip_top(recorded['observation']):audit['control_mismatches'].append([stage,tid,'observation'])
            extracted=saved['extracted'];score=dict(all_correct_accepted=False,any_wrong_accepted=False)
            if extracted['status']=='ok':
                if cohort=='synthetic':
                    if prepared is None:prepared=prepare(task,root)
                    score=probe(extracted['contract'],prepared)
                else:score=dict(all_correct_accepted=verify(extracted['contract'],before,ref.df_out)[0].status=='PASS',any_wrong_accepted=verify(extracted['contract'],before,before)[0].status=='PASS')
            for k,v in score.items():
                if r.get(k)!=v:audit['score_mismatches'].append([stage,tid,k])
            case['stages'][stage]=dict(status=r['parse_status'],correct=r['all_correct_accepted'],wrong=r['any_wrong_accepted'],tools=saved['tool_calls'],contract=extracted.get('contract'),score=score)
        audit['cases'][tid]=case
        if index%10==0:print('CPU replay',index,'/100 for both checkpoints',flush=True)
    for stage,records in all_stages.items():
        values=list(records.values());stats=dict(correct=sum(r['all_correct_accepted'] for r in values),wrong=sum(r['any_wrong_accepted'] for r in values),joint=sum(r['all_correct_accepted'] and not r['any_wrong_accepted'] for r in values),valid=sum(r['parse_status']=='ok' for r in values),
           tokens=sum(r['total_tokens'] for r in values),llm_calls=sum(r['parser']['llm_calls'] for r in values),seconds=sum(r['parser']['seconds'] for r in values),tool_attempts=sum(r['tool_attempts'] for r in values),tool_failures=sum(r['tool_failures'] for r in values),
           tasks_with_tools=sum(r['tool_attempts']>0 for r in values),correct_with_tools=sum(r['tool_attempts']>0 and r['all_correct_accepted'] for r in values),placeholder_calls=sum(t['args']=={'column':'__unknown_column__'} for r in values for t in r['parser']['tool_calls']),
           by_pattern={})
        for pattern in sorted({c['pattern'] for c in audit['cases'].values()}):
            group=[records[tid] for tid,c in audit['cases'].items() if c['pattern']==pattern]
            stats['by_pattern'][pattern]=dict(tasks=len(group),correct=sum(r['all_correct_accepted'] for r in group),wrong=sum(r['any_wrong_accepted'] for r in group),joint=sum(r['all_correct_accepted'] and not r['any_wrong_accepted'] for r in group))
        audit['stats'][stage]=stats
    chosen=stages[-1];audit['paired_vs_base']=dict(gained=[tid for tid in expected if rows[chosen][tid]['all_correct_accepted'] and not baseline[tid]['all_correct_accepted']],lost=[tid for tid in expected if baseline[tid]['all_correct_accepted'] and not rows[chosen][tid]['all_correct_accepted']])
    with zipfile.ZipFile(ROOT/'kaggle_outputs/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval.zip') as z:
        sf={r['task_id']:r for n in z.namelist() if n.startswith('tasks/') and (r:=json.loads(z.read(n)))}
    audit['paired_vs_scopefix']=dict(gained=[tid for tid in expected if rows[chosen][tid]['all_correct_accepted'] and not sf[tid]['all_correct_accepted']],lost=[tid for tid in expected if sf[tid]['all_correct_accepted'] and not rows[chosen][tid]['all_correct_accepted']])
    training=[json.loads(l) for l in (data/'sft_train.jsonl').read_text().splitlines()]
    audit['training_artifact']=dict(recovery_conversations=sum(r['recovery'] for r in training),recovery_first_assistant_placeholder=sum(r['recovery'] and next(m['content'] for m in r['messages'] if m['role']=='assistant')=='{"tool": "profile_column", "args": {"column": "__unknown_column__"}}' for r in training),
      explanation='The all-assistant-turn loss also supervises intentionally bad recovery-prefix actions. This gives the exact placeholder positive training weight; the same first action appears on all 16 statistic development cases. This is an undesirable training-design artifact, not necessary tool discovery.')
    write(OUT/'audit.json',audit)
    print('SAVED',OUT)
    print('CHECKS',{k:audit[k] for k in ['notebook_source_matches','notebook_errors','summary_mismatches','score_mismatches','control_mismatches']})
    print('PAIRED',audit['paired_vs_scopefix'])

if __name__=='__main__':main()
