"""Audit the supplied SF1 training bundle without GPU calls or runtime changes."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import nbformat
import pandas as pd
from finetune.selective_facts.context import messages,selective_facts,FactUnavailable
from finetune.selective_facts.evaluate import development_tasks,summarize,eligible
from finetune.selective_facts.adapter import validate_adapter
from finetune.selective_facts.provenance import digest,write
from eval.contract_quality.harness import prepare,probe
from igt.agents.intent_parser import extract_contract
from igt.sandbox import safe_run
from igt.orchestrator import Limits
from igt.verifier import verify
from eval.equivalence import outputs_equal

OUT=ROOT/'results/selective_facts_analysis/v1'

def main():
    source=ROOT/'kaggle_outputs/training/selective_facts_v1/selective_facts_1_5b.zip'
    with zipfile.ZipFile(source) as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Invalid ZIP')
        seal=json.loads(z.read('bundle.json'))
        if not seal['complete'] or set(seal['sha256'])!=set(z.namelist())-{'bundle.json'}:raise ValueError('Invalid bundle coverage')
        for name,sha in seal['sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=sha:raise ValueError('Bundle seal mismatch: '+name)
        completion=json.loads(z.read('completion.json'));selection=json.loads(z.read('selection.json'));info=json.loads(z.read('run_info.json'))
        if not completion['complete'] or completion['smoke']:raise ValueError('Incomplete or smoke run')
        stages=['baseline']+['semantic/'+r['name'] for r in selection['all_candidates']]
        rows={stage:{r['task_id']:r for name in z.namelist() if name.startswith(stage+'/tasks/') and (r:=json.loads(z.read(name)))} for stage in stages}
        supplied={stage:json.loads(z.read(stage+'/summary.json')) for stage in stages}
        # Only the delivered adapter is extracted, into a temporary audit directory.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='sf1_adapter_audit_') as d:
            folder=Path(d)
            for name in z.namelist():
                if name.startswith('adapter/'):
                    relative=Path(name).relative_to('adapter')
                    if len(relative.parts)!=1:raise ValueError('Unexpected adapter path')
                    (folder/relative).write_bytes(z.read(name))
            adapter_validation=validate_adapter(folder)
        logs=[json.loads(l) for l in z.read('train_log.jsonl').decode().splitlines()]
        events=[json.loads(l) for l in z.read('gpu_events.jsonl').decode().splitlines()]
    notebook=nbformat.read(ROOT/'notebooks/executed/training/selective_facts_v1/selective_facts_train_executed.ipynb',as_version=4)
    template=nbformat.read(ROOT/'notebooks/kaggle/selective_facts/selective_facts_train.ipynb',as_version=4)
    tasks=development_tasks();expected={t['id'] for t,_,_ in tasks}
    audit=dict(bundle_sha256=digest(source),complete=completion,selection=selection,adapter_validation=adapter_validation,
      source_mismatches=[n for n,h in info['source_hashes'].items() if digest(ROOT/n)!=h],
      data_manifest_matches=digest(ROOT/'finetune/data/selective_facts_v1/manifest.json')==info['data_manifest_sha256'],
      notebook_source_matches=[c.source for c in notebook.cells]==[c.source for c in template.cells],
      notebook_errors=[dict(cell=i,error=o.get('evalue')) for i,c in enumerate(notebook.cells) for o in c.get('outputs',[]) if o.output_type=='error'],
      message_mismatches=[],extraction_mismatches=[],score_mismatches=[],summary_mismatches=[],cases={},by_pattern={},routing_counterfactual=[],
      training=dict(seconds=info['training_seconds'],steps=info['steps'],train_loss=info['train_loss'],final_eval=info['final_eval'],validation_losses=[r for r in logs if 'eval_loss' in r],versions=info['versions'],config=info['config'],token_lengths=info['token_lengths'],attention_events=dict(collections.Counter(e.get('event') for e in events))))
    original_spec=json.loads((ROOT/'eval/stage2_independent_v2/adapter.json').read_text())['sha256']
    audit['baseline_adapter_matches']=all(supplied['baseline']['provenance']['adapter_hashes'].get(n)==h for n,h in original_spec.items())
    if not audit['baseline_adapter_matches']:raise ValueError('Wrong original adapter control')
    for stage in stages:
        if set(rows[stage])!=expected:raise ValueError('Task coverage differs: '+stage)
        if summarize(list(rows[stage].values()))!=supplied[stage]['summary']:audit['summary_mismatches'].append(stage)
        provenance=supplied[stage]['provenance']
        for name,sha in provenance['inputs'].items():
            if digest(ROOT/name)!=sha:raise ValueError('Input changed: '+name)
        for name,sha in provenance['original_task_hashes'].items():
            if digest(ROOT/name)!=sha:raise ValueError('Development request changed: '+name)
    for index,(task,cohort,root) in enumerate(tasks,1):
        tid=task['id'];before=pd.read_csv(root/task['data_file']);prepared=None
        if cohort=='original':
            ref=safe_run(task['reference_code'],before,**Limits().sandbox())
            if not ref.ran_ok:raise ValueError(ref.error)
            if outputs_equal(ref.df_out,before)[0]:raise ValueError('Wrong no-op not distinct')
        case=dict(request=task['request'],cohort=cohort,pattern=task.get('pattern',task.get('transformation_type')),stages={})
        for stage in stages:
            r=rows[stage][tid];detail=dict(parse_status=r['parse_status'],correct=r['all_correct_accepted'],wrong=r['any_wrong_accepted'],error=r.get('error'))
            try:msg=messages(task,before,original=stage=='baseline')
            except FactUnavailable as exc:
                if r['parse_status']!='fact_unavailable' or r.get('error')!=str(exc):audit['message_mismatches'].append([stage,tid])
                detail['error']=str(exc)
                if stage==stages[-1]:
                    # Diagnostic only: remove semantically neutral wording; do not generate a new prediction.
                    request=task['request'].replace('Ignore missing values when computing statistics. ','')
                    try:facts=selective_facts(request,before)
                    except FactUnavailable as e:facts=dict(error=str(e))
                    audit['routing_counterfactual'].append(dict(task_id=tid,original_error=str(exc),neutral_sentence_removed=request,facts=facts))
            else:
                if msg!=r.get('messages'):audit['message_mismatches'].append([stage,tid])
                extracted=extract_contract(r['raw_output'])
                if extracted['status']!=r['parse_status'] or extracted.get('contract')!=r.get('predicted_contract'):audit['extraction_mismatches'].append([stage,tid])
                detail['error']=extracted.get('error');detail['contract']=extracted.get('contract')
                if extracted['status']=='ok':
                    if cohort=='synthetic':
                        if prepared is None:prepared=prepare(task,root)
                        score=probe(extracted['contract'],prepared)
                    else:score=dict(all_correct_accepted=verify(extracted['contract'],before,ref.df_out)[0].status=='PASS',any_wrong_accepted=verify(extracted['contract'],before,before)[0].status=='PASS')
                    for key,value in score.items():
                        if r.get(key)!=value:audit['score_mismatches'].append([stage,tid,key])
                    if cohort=='original':detail['reference_verdict']=verify(extracted['contract'],before,ref.df_out)[0].__dict__
                    else:detail['positive_verdicts']=score['positive_verdicts']
            case['stages'][stage]=detail
        audit['cases'][tid]=case
        if index%10==0:print(f'CPU replay {index}/100',flush=True)
    chosen=stages[-1]
    audit['summaries']={stage:supplied[stage]['summary'] for stage in stages}
    audit['paired']={}
    for cohort in ['synthetic','original']:
        ids=[t['id'] for t,c,_ in tasks if c==cohort]
        audit['paired'][cohort]=dict(gained=[i for i in ids if not rows['baseline'][i]['all_correct_accepted'] and rows[chosen][i]['all_correct_accepted']],lost=[i for i in ids if rows['baseline'][i]['all_correct_accepted'] and not rows[chosen][i]['all_correct_accepted']])
    for pattern in sorted({c['pattern'] for c in audit['cases'].values()}):
        cases=[c for c in audit['cases'].values() if c['pattern']==pattern]
        audit['by_pattern'][pattern]={stage:dict(tasks=len(cases),correct=sum(c['stages'][stage]['correct'] for c in cases),wrong=sum(c['stages'][stage]['wrong'] for c in cases),parse=dict(collections.Counter(c['stages'][stage]['parse_status'] for c in cases))) for stage in stages}
    audit['eligibility_recomputed']=eligible(supplied[chosen]['summary'],supplied['baseline']['summary'])
    if audit['eligibility_recomputed']!=selection['eligible']:raise ValueError('Selection rule mismatch')
    write(OUT/'audit.json',audit)
    print('AUDIT',OUT)
    print('CHECKS',json.dumps({k:audit[k] for k in ['source_mismatches','data_manifest_matches','notebook_source_matches','notebook_errors','message_mismatches','extraction_mismatches','score_mismatches','summary_mismatches','eligibility_recomputed']}))
    print('PATTERNS',json.dumps(audit['by_pattern']))
    print('PAIRED',json.dumps(audit['paired']))

if __name__=='__main__':main()
