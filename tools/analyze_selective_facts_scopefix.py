"""Verify new scope-fixed results; replay new cases and bind unchanged cases to prior CPU audit."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import nbformat
import pandas as pd
from finetune.selective_facts_scopefix.context import messages,selective_facts
from finetune.selective_facts_scopefix.comparison import compare
from finetune.selective_facts_scopefix.adapter import SPEC
from finetune.selective_facts.evaluate import development_tasks,summarize
from finetune.selective_facts.provenance import digest,write
from eval.contract_quality.harness import prepare,probe
from igt.agents.intent_parser import extract_contract

OUT=ROOT/'results/selective_facts_analysis/scopefix1'

def sealed(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Invalid ZIP')
        seal=json.loads(z.read('bundle.json'))
        if not seal['complete'] or set(seal['sha256'])!=set(z.namelist())-{'bundle.json'}:raise ValueError('Incomplete coverage')
        for name,sha in seal['sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=sha:raise ValueError('Changed bundle member: '+name)
        return {n:json.loads(z.read(n)) for n in z.namelist() if n.endswith('.json') and n!='bundle.json'}

def main():
    source=ROOT/'kaggle_outputs/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval.zip';files=sealed(source)
    summary=files['summary.json'];completion=files['completion.json']
    if not completion['complete'] or completion['smoke'] or completion['retrained']:raise ValueError('Unexpected run type')
    prior_source=ROOT/'kaggle_outputs/training/selective_facts_v1/selective_facts_1_5b.zip'
    if digest(prior_source)!=SPEC['source_bundle_sha256']:raise ValueError('Wrong historical training bundle')
    prior=sealed(prior_source);stage='semantic/'+SPEC['selected']
    old={r['task_id']:r for n,r in prior.items() if n.startswith(stage+'/tasks/')}
    rows={r['task_id']:r for n,r in files.items() if n.startswith('tasks/')};tasks=development_tasks()
    if set(rows)!={t['id'] for t,_,_ in tasks}:raise ValueError('Incomplete coverage')
    if summarize(list(rows.values()))!=summary['summary']:raise ValueError('Summary differs from records')
    regenerated=compare([rows[t['id']] for t,_,_ in tasks])
    if regenerated!=files['comparison.json']:raise ValueError('Historical comparison mismatch')
    prov=summary['provenance']
    for group in ['runtime','inputs','original_task_hashes']:
        for n,sha in prov[group].items():
            if digest(ROOT/n)!=sha:raise ValueError('Changed runtime/input: '+n)
    if prov['protocol']!='p1-v2+sf1-scopefix1' or prov['adapter_trained_protocol']!='p1-v2+sf1':raise ValueError('Protocol mismatch')
    if any(prov['adapter_hashes'].get(n)!=sha for n,sha in SPEC['sha256'].items()):raise ValueError('Wrong adapter seals')
    old_prov=prior[stage+'/summary.json']['provenance']
    if prov['inputs']!=old_prov['inputs'] or prov['original_task_hashes']!=old_prov['original_task_hashes']:raise ValueError('Historical inputs changed')
    prior_audit_path=ROOT/'results/selective_facts_analysis/v1/audit.json';prior_audit=json.loads(prior_audit_path.read_text())
    if prior_audit['bundle_sha256']!=digest(prior_source) or any(prior_audit[k] for k in ['source_mismatches','message_mismatches','extraction_mismatches','score_mismatches','summary_mismatches']):raise ValueError('Prior CPU audit not clean')
    for n,sha in prior['run_info.json']['source_hashes'].items():
        if digest(ROOT/n)!=sha:raise ValueError('Prior audited verifier/runtime changed')
    book=nbformat.read(ROOT/'notebooks/executed/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval_colab_executed.ipynb',as_version=4)
    audit=dict(bundle_sha256=digest(source),completion=completion,summary=summary['summary'],comparison=regenerated,
      notebook_errors=[dict(cell=i,error=o.get('evalue')) for i,c in enumerate(book.cells) for o in c.get('outputs',[]) if o.output_type=='error'],
      launcher='Colab adaptation; source bundle and executed Python runtime hashes verified rather than requiring Kaggle launcher equality.',
      environment_limit='The Colab launcher uses its installed Torch/CUDA build, not the Kaggle Torch assertion. colab_environment.json is written to Drive but is not present in this result ZIP; exact Torch/GPU identity cannot be verified from the supplied ZIP.',
      historical_audit_sha256=digest(prior_audit_path),unchanged_prior_audit_bound=[],new_cpu_replays=[],message_mismatches=[],facts_mismatches=[],extraction_mismatches=[],score_mismatches=[],cases={},by_pattern={})
    for task,cohort,root in tasks:
        tid=task['id'];r=rows[tid];previous=old[tid];before=pd.read_csv(root/task['data_file'])
        facts=selective_facts(task['request'],before)
        if r['messages']!=messages(task,before):audit['message_mismatches'].append(tid)
        if r['facts']!=facts:audit['facts_mismatches'].append(tid)
        extracted=extract_contract(r['raw_output'])
        if extracted['status']!=r['parse_status'] or extracted.get('contract')!=r.get('predicted_contract'):audit['extraction_mismatches'].append(tid)
        if previous['parse_status']=='fact_unavailable':
            if cohort!='synthetic' or extracted['status']!='ok':raise ValueError('Unsolved repaired case')
            score=probe(extracted['contract'],prepare(task,root))
            for k,v in score.items():
                if r.get(k)!=v:audit['score_mismatches'].append([tid,k])
            audit['new_cpu_replays'].append(tid)
            print('CPU replay repaired case',tid,score['all_correct_accepted'],score['any_wrong_accepted'],flush=True)
        else:
            if r['messages']!=previous['messages'] or r['raw_output']!=previous['raw_output'] or r['predicted_contract']!=previous['predicted_contract']:raise ValueError('Nonblocked prompt/prediction changed: '+tid)
            for k in ['parse_status','all_correct_accepted','any_wrong_accepted','positive_statuses','positive_verdicts','negative_statuses']:
                if r.get(k)!=previous.get(k):audit['score_mismatches'].append([tid,k])
            audited=prior_audit['cases'][tid]['stages'][stage]
            if audited['correct']!=r['all_correct_accepted'] or audited['wrong']!=r['any_wrong_accepted']:raise ValueError('Prior audit label mismatch')
            audit['unchanged_prior_audit_bound'].append(tid)
        case=dict(request=task['request'],cohort=cohort,pattern=task.get('pattern',task.get('transformation_type')),facts=facts,status=r['parse_status'],correct=r['all_correct_accepted'],wrong=r['any_wrong_accepted'],error=extracted.get('error'),contract=extracted.get('contract'),positive_verdicts=r.get('positive_verdicts'))
        audit['cases'][tid]=case
    for pattern in sorted({c['pattern'] for c in audit['cases'].values()}):
        group=[c for c in audit['cases'].values() if c['pattern']==pattern]
        audit['by_pattern'][pattern]=dict(tasks=len(group),correct=sum(c['correct'] for c in group),wrong=sum(c['wrong'] for c in group),invalid=sum(c['status']!='ok' for c in group))
    audit['totals']=dict(correct=sum(r['all_correct_accepted'] for r in rows.values()),wrong=sum(r['any_wrong_accepted'] for r in rows.values()),valid=sum(r['parse_status']=='ok' for r in rows.values()),
      facts_delivered=sum(bool(r['facts']) for r in rows.values()),tokens=sum(r['prompt_tokens']+r['output_tokens'] for r in rows.values()),inference_seconds=sum(r.get('seconds',0) for r in rows.values()))
    write(OUT/'audit.json',audit)
    print('SAVED',OUT)
    print('CHECKS',{k:len(audit[k]) for k in ['notebook_errors','message_mismatches','facts_mismatches','extraction_mismatches','score_mismatches','new_cpu_replays','unchanged_prior_audit_bound']})

if __name__=='__main__':main()
