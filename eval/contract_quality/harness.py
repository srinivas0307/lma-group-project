"""Sandboxed suite validation and semantic parser scoring; selection requires explicit opt-in."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import pandas as pd
from igt.sandbox import safe_run
from igt.verifier import verify
from igt.agents.intent_parser import extract_contract
from eval.equivalence import outputs_equal
from .build_suite import SUITE

def load_tasks(suite=SUITE, split='dev', unlock_selection=False):
    if split=='selection' and not unlock_selection:
        raise ValueError('Selection is locked. Freeze the candidate and document the one-time evaluation before opting in.')
    suite=Path(suite)
    manifest=json.loads((suite/'manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        if hashlib.sha256((suite/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Suite hash mismatch: {name}')
    return [json.loads(s) for s in (suite/f'{split}.jsonl').read_text().splitlines()]

def run(code,before):
    result=safe_run(code,before,timeout_s=30,mem_gb=2,cpu_s=20)
    if not result.ran_ok:
        raise ValueError(f'Candidate execution failed: {result.status}: {result.error}')
    return result.df_out

def prepare(task,suite=SUITE):
    before=pd.read_csv(Path(suite)/task['data_file'])
    reference=run(task['reference_code'],before)
    positives=[reference]+[run(c,before) for c in task['positive_codes'] if c!=task['reference_code']]
    negatives=[(n['family'],run(n['code'],before)) for n in task['negative_codes']]
    for pos in positives:
        if not outputs_equal(reference,pos)[0]:raise ValueError(f"{task['id']}: alternate is not equivalent")
    for family,neg in negatives:
        if outputs_equal(reference,neg)[0]:raise ValueError(f"{task['id']}: {family} mutant is equivalent")
    return before,positives,negatives

def probe(contract, prepared):
    before,positives,negatives=prepared
    pos=[verify(contract,before,p)[0].status for p in positives]
    neg=[dict(family=f,status=verify(contract,before,n)[0].status) for f,n in negatives]
    return dict(positive_statuses=pos,negative_statuses=neg,all_correct_accepted=all(s=='PASS' for s in pos),
                any_wrong_accepted=any(n['status']=='PASS' for n in neg))

def validate(suite=SUITE):
    rows=[]
    for split in ['dev','selection']:
        for t in load_tasks(suite,split,True):
            try:
                r=probe(t['gold_contract'],prepare(t,suite))
                r['valid']=r['all_correct_accepted'] and not r['any_wrong_accepted']
            except Exception as e:r=dict(valid=False,error=str(e))
            rows.append(dict(task_id=t['id'],split=split,**r))
            if len(rows)%20==0:print(f'Validated {len(rows)}/120',flush=True)
    return dict(kind='automated_draft_validation',human_reviewed=False,valid=all(r['valid'] for r in rows),rows=rows)

def score(records,suite=SUITE):
    tasks={t['id']:t for t in load_tasks(suite)}
    seen=set();groups=collections.defaultdict(list); prepared={}
    for record in records:
        tid=record['task_id'];variant=record['variant'];key=(variant,tid)
        if key in seen or tid not in tasks:raise ValueError(f'Duplicate or unexpected record {key}')
        seen.add(key)
        parsed=extract_contract(record['raw_output'])
        result=dict(task_id=tid,family=tasks[tid]['family'],parse_status=parsed['status'])
        if parsed['status']=='ok':
            if tid not in prepared:prepared[tid]=prepare(tasks[tid],suite)
            result.update(probe(parsed['contract'],prepared[tid]))
        else:result.update(all_correct_accepted=False,any_wrong_accepted=False)
        result['prompt_tokens']=record.get('prompt_tokens',0);result['output_tokens']=record.get('output_tokens',0)
        groups[variant].append(result)
    summary={}
    for name,rows in groups.items():
        summary[name]=dict(records=len(rows),expected=len(tasks),complete=len(rows)==len(tasks),
            correct_accepted=sum(r['all_correct_accepted'] for r in rows),
            tasks_with_wrong_acceptance=sum(r['any_wrong_accepted'] for r in rows),
            invalid=sum(r['parse_status']!='ok' for r in rows),
            joint_success=sum(r['all_correct_accepted'] and not r['any_wrong_accepted'] for r in rows),
            tokens=sum(r['prompt_tokens']+r['output_tokens'] for r in rows))
    paired={}
    for left in groups:
        for right in groups:
            if left>=right:continue
            l={r['task_id']:r for r in groups[left]};r={r['task_id']:r for r in groups[right]}
            ids=sorted(l.keys()&r.keys())
            paired[left+'__vs__'+right]=dict(matched_tasks=len(ids),**{
                metric:dict(left_only=sum(l[i][metric] and not r[i][metric] for i in ids),
                            right_only=sum(r[i][metric] and not l[i][metric] for i in ids))
                for metric in ['all_correct_accepted','any_wrong_accepted']})
    for name,rows in groups.items():
        families=collections.defaultdict(lambda:dict(probes=0,accepted=0))
        for row in rows:
            for n in row.get('negative_statuses',[]):
                families[n['family']]['probes']+=1
                families[n['family']]['accepted']+=n['status']=='PASS'
        summary[name]['wrong_candidates_by_family']=dict(families)
    return dict(kind='exploratory_synthetic_draft',summary=summary,paired=paired,rows=dict(groups),
                note='Invalid contracts count as withheld, not as successful rejection; no model winner is declared automatically.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=score([json.loads(l) for l in a.predictions.read_text().splitlines()]) if a.predictions else validate()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    if not a.predictions and not result['valid']:raise SystemExit('Draft validation failed; inspect report')

if __name__=='__main__':main()
