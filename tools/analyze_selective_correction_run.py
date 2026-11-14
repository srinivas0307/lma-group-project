"""Audit fresh correction experiment by exact policy replay and prior CPU-score bindings."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
import nbformat
import pandas as pd
from analyze_selective_facts_scopefix import sealed
from finetune.selective_facts.evaluate import development_tasks
from finetune.selective_facts_correction.parser import attempt_correction,ARMS
from finetune.selective_facts_correction.evaluate import comparison
from finetune.selective_facts_correction.artifacts import digest,write

OUT=ROOT/'results/selective_facts_analysis/correction1/completed'

def main():
    source=ROOT/'kaggle_outputs/parsing/selective_facts_correction1/selective_facts_correction_eval.zip';files=sealed(source)
    completion=files['completion.json'];summary=files['summary.json'];info=files['run_info.json']
    if not completion['complete'] or completion['smoke'] or completion['retrained']:raise ValueError('Wrong run type')
    for group in ['runtime','input_hashes']:
        for n,sha in info[group].items():
            if digest(ROOT/n)!=sha:raise ValueError('Changed source/input: '+n)
    prior_path=ROOT/'kaggle_outputs/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval.zip';prior=sealed(prior_path)
    prior_rows={r['task_id']:r for n,r in prior.items() if n.startswith('tasks/')}
    if info['adapter_validation']!=prior['summary.json']['provenance']['adapter_validation']:raise ValueError('Adapter differs')
    old_audit_path=ROOT/'results/selective_facts_analysis/scopefix1/audit.json';old_audit=json.loads(old_audit_path.read_text())
    if old_audit['bundle_sha256']!=digest(prior_path) or any(old_audit[k] for k in ['message_mismatches','facts_mismatches','extraction_mismatches','score_mismatches']):raise ValueError('Prior audit binding failed')
    offline_path=ROOT/'results/selective_facts_analysis/correction1/offline_audit.json';offline=json.loads(offline_path.read_text())
    if offline['source_bundle_sha256']!=digest(prior_path) or offline['checks_sha256']!=digest(ROOT/'finetune/selective_facts_correction/checks.py'):raise ValueError('Offline normalization binding failed')
    rows={r['task_id']:r for n,r in files.items() if n.startswith('tasks/')};tasks=development_tasks()
    if set(rows)!={t['id'] for t,_,_ in tasks}:raise ValueError('Incomplete coverage')
    task_sha=__import__('hashlib').sha256(json.dumps([t for t,_,_ in tasks],sort_keys=True).encode()).hexdigest()
    if task_sha!=info['task_sha256']:raise ValueError('Development task changed')
    regenerated=comparison([rows[t['id']] for t,_,_ in tasks])
    if regenerated!=files['comparison.json'] or regenerated['current']!=summary['arms']:raise ValueError('Comparison or summary mismatch')
    book=nbformat.read(ROOT/'notebooks/executed/parsing/selective_facts_correction1/selective_facts_correction_colab_executed.ipynb',as_version=4)
    template=nbformat.read(ROOT/'notebooks/colab/selective_facts_correction/selective_facts_correction_colab.ipynb',as_version=4)
    audit=dict(bundle_sha256=digest(source),completion=completion,summary=summary,comparison=regenerated,environment=info['environment'],
      notebook_source_matches=[c.source for c in book.cells]==[c.source for c in template.cells],notebook_errors=[dict(cell=i,error=o.get('evalue')) for i,c in enumerate(book.cells) for o in c.get('outputs',[]) if o.output_type=='error'],
      previous_cpu_audit_sha256=digest(old_audit_path),normalization_cpu_audit_sha256=digest(offline_path),policy_mismatches=[],score_mismatches=[],unchanged_control_predictions=[],correction_details={},bound_cases=[])
    for task,cohort,root in tasks:
        tid=task['id'];r=rows[tid];before=pd.read_csv(root/task['data_file']);saved=r['trace']['replies']
        class Replay:
            def __init__(self):self.i=0
            def complete(self,messages,*args):
                reply=saved[self.i]
                if messages!=reply['messages']:audit['policy_mismatches'].append([tid,'model_input',self.i])
                self.i+=1;return reply['raw_output'],reply['prompt_tokens'],reply['output_tokens']
        replay=attempt_correction(dict(request=task['request'],data_file=task['data_file']),before,Replay())
        for a in ARMS:
            if replay['variants'][a]!={k:v for k,v in r['variants'][a].items() if k!='score'}:audit['policy_mismatches'].append([tid,a])
        for key in set(replay['trace'])|set(r['trace']):
            if key!='seconds' and replay['trace'].get(key)!=r['trace'].get(key):audit['policy_mismatches'].append([tid,'trace',key])
        if saved[0]['raw_output']!=prior_rows[tid]['raw_output'] or saved[0]['messages']!=prior_rows[tid]['messages']:raise ValueError('Fresh control prediction differs from prior: '+tid)
        audit['unchanged_control_predictions'].append(tid)
        for a in ARMS:
            candidate=r['variants'][a];s=candidate['score']
            if a=='control':expected={k:prior_rows[tid][k] for k in ['all_correct_accepted','any_wrong_accepted']}
            elif a=='normalized':expected=dict(all_correct_accepted=offline['cases'][tid]['normalized_correct'],any_wrong_accepted=offline['cases'][tid]['normalized_wrong'])
            elif r['trace']['attempted']:expected=dict(all_correct_accepted=False,any_wrong_accepted=False)
            else:expected={k:r['variants']['normalized']['score'][k] for k in ['all_correct_accepted','any_wrong_accepted']}
            # In this run all attempted corrections are rejected by input/schema diagnostics.
            if a=='corrected' and r['trace']['attempted'] and candidate['status']!='correction_failed':raise ValueError('New accepted revision requires fresh CPU semantic replay')
            for k,v in expected.items():
                if s[k]!=v:audit['score_mismatches'].append([tid,a,k])
        if r['trace']['attempted']:
            audit['correction_details'][tid]=dict(identical_reply=saved[0]['raw_output']==saved[1]['raw_output'],initial_issues=r['trace']['initial_issues'],remaining_issues=r['trace']['remaining_issues'],status=r['variants']['corrected']['status'])
        audit['bound_cases'].append(tid)
    calls=sum(r['trace']['llm_calls'] for r in rows.values());tokens=sum(r['trace']['prompt_tokens']+r['trace']['output_tokens'] for r in rows.values())
    if calls!=summary['actual_model_calls'] or tokens!=summary['actual_tokens']:raise ValueError('Actual cost mismatch')
    audit['totals']=dict(control_correct=85,normalized_correct=88,corrected_correct=88,correction_attempts=len(audit['correction_details']),identical_correction_replies=sum(c['identical_reply'] for c in audit['correction_details'].values()),
       calls=calls,tokens=tokens,inference_seconds=sum(r['trace']['seconds'] for r in rows.values()),additional_tokens=tokens-166045)
    write(OUT/'audit.json',audit)
    print('SAVED',OUT)
    print('CHECKS',{k:audit[k] if isinstance(audit[k],bool) else len(audit[k]) for k in ['notebook_source_matches','notebook_errors','policy_mismatches','score_mismatches','unchanged_control_predictions','bound_cases']})
    print('TOTALS',audit['totals'])
if __name__=='__main__':main()
