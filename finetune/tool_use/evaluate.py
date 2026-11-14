"""Free-generation semantic scoring on OPEN development tasks, never training or selection."""
import argparse
import collections
import dataclasses
import json
import time
from pathlib import Path
import pandas as pd
import yaml
from igt.agents.intent_parser import extract_contract
from igt.verifier import verify
from igt.sandbox import safe_run
from igt.sandbox.environment import sandbox_environment
from igt.orchestrator import Limits
from eval.equivalence import outputs_equal
from eval.contract_quality.harness import prepare,probe
from .config import PROTOCOL,MODELS
from .parser import parse_fresh
from finetune.selective_facts.provenance import ROOT,write,digest
from .provenance import hashes

def development_tasks(smoke=False):
    suite=ROOT/'benchmark/contract_quality_v1'
    manifest=json.loads((suite/'manifest.json').read_text())
    if digest(suite/'dev.jsonl')!=manifest['sha256']['dev.jsonl']:raise ValueError('Development manifest changed')
    synthetic=[json.loads(line) for line in (suite/'dev.jsonl').read_text().splitlines()]
    for name in {t['data_file'] for t in synthetic}:
        if digest(suite/name)!=manifest['sha256'][name]:raise ValueError('Development CSV changed')
    originals=[t for p in sorted((ROOT/'benchmark/tasks').glob('*/*.yaml')) if (t:=yaml.safe_load(p.read_text()))['split']=='dev']
    if smoke:
        synthetic=[next(t for t in synthetic if t['pattern']=='fill_0'),next(t for t in synthetic if t['pattern']=='dedup_0')]
        originals=[]
    return [(t,'synthetic',suite) for t in synthetic]+[(t,'original',ROOT) for t in originals]

def summarize(rows):
    results={}
    for cohort in ['synthetic','original']:
        group=[r for r in rows if r['cohort']==cohort]
        results[cohort]=dict(tasks=len(group),correct_accepted=sum(r['all_correct_accepted'] for r in group),
            wrong_accepted=sum(r['any_wrong_accepted'] for r in group),joint_success=sum(r['all_correct_accepted'] and not r['any_wrong_accepted'] for r in group),
            invalid=sum(r['parse_status']!='ok' for r in group),tokens=sum(r['total_tokens'] for r in group))
    return results

def eligible(candidate,baseline):
    # Screening on finite open-development data, not a statistical safety guarantee.
    return (all(candidate[c]['tasks']==baseline[c]['tasks'] and candidate[c]['tasks']>0 and
                candidate[c]['correct_accepted']>=baseline[c]['correct_accepted'] and
                candidate[c]['wrong_accepted']<=baseline[c]['wrong_accepted'] for c in baseline)
            and sum(candidate[c]['joint_success'] for c in baseline)>sum(baseline[c]['joint_success'] for c in baseline))

def evaluate(adapter,out,original=False,packages=None,smoke=False,size='1_5b'):
    out=Path(out)
    if out.exists():raise ValueError('Evaluation output already exists')
    out.mkdir(parents=True)
    tasks=development_tasks(smoke)
    import torch
    from .backend import Backend
    from finetune.selective_facts.attention import configure_attention
    configure_attention(torch,lambda event:None)
    backend=Backend(size,adapter)
    rows=[]
    try:
        with sandbox_environment(packages):
            for index,(task,cohort,root) in enumerate(tasks,1):
                before=pd.read_csv(root/task['data_file'])
                record=dict(task_id=task['id'],cohort=cohort,request=task['request'],total_tokens=0,
                            all_correct_accepted=False,any_wrong_accepted=False,parse_status='fact_unavailable')
                parsed=parse_fresh(task,before,backend,'p1-v2')
                extracted=parsed['extracted']
                record.update(parser=parsed,total_tokens=parsed['tokens'],
                              parse_status=extracted['status'],predicted_contract=extracted.get('contract'),
                              tool_attempts=parsed['tool_attempts'],tool_failures=parsed['tool_failures'])
                if extracted['status']=='ok':
                    if cohort=='synthetic':record.update(probe(extracted['contract'],prepare(task,root)))
                    else:
                        reference=safe_run(task['reference_code'],before,**Limits().sandbox())
                        if not reference.ran_ok:raise RuntimeError(f'Reference failed: {task["id"]}: {reference.error}')
                        if outputs_equal(reference.df_out,before)[0]:raise ValueError('No-op is not a negative')
                        record.update(all_correct_accepted=verify(extracted['contract'],before,reference.df_out)[0].status=='PASS',
                                      any_wrong_accepted=verify(extracted['contract'],before,before)[0].status=='PASS')
                rows.append(record);write(out/'tasks'/f'{cohort}_{task["id"]}.json',record)
                print(f'DEV {index}/{len(tasks)} {task["id"]}: {record["parse_status"]}',flush=True)
        summary=summarize(rows)
        for cohort,stats in summary.items():
            group=[r for r in rows if r['cohort']==cohort]
            stats.update(tasks_with_tools=sum(r['tool_attempts']>0 for r in group),tool_attempts=sum(r['tool_attempts'] for r in group),tool_failures=sum(r['tool_failures'] for r in group))
        inputs={str((root/task['data_file']).relative_to(ROOT)):digest(root/task['data_file']) for task,_,root in tasks}
        provenance=dict(runtime=hashes(),inputs=inputs,synthetic_manifest=digest(ROOT/'benchmark/contract_quality_v1/manifest.json'),
            original_task_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'benchmark/tasks').glob('*/*.yaml') if yaml.safe_load(p.read_text())['split']=='dev'},
            adapter_hashes={p.name:digest(p) for p in (Path(adapter).iterdir() if adapter else []) if p.is_file() and p.suffix in ['.json','.safetensors','.txt']},
            protocol=PROTOCOL,size=size,base=MODELS[size],arm='trained' if adapter else 'base')
        result=dict(complete=True,smoke=smoke,summary=summary,provenance=provenance,limitations=['Open development data; synthetic draft not independently reviewed.','Original tasks have only a wrong-no-op probe; synthetic tasks have targeted wrong outputs.'])
        write(out/'summary.json',result);return result
    finally:backend.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--size',choices=MODELS,default='1_5b');p.add_argument('--packages',type=Path);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    from finetune.selective_facts.train import export
    try:evaluate(a.adapter,a.out,packages=a.packages,smoke=a.smoke,size=a.size)
    finally:
        if a.out.exists():export(a.out,(a.out/'summary.json').exists())
if __name__=='__main__':main()
