"""No model calls: test narrow normalization and input-only trigger coverage on saved dev replies."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import pandas as pd
from finetune.selective_facts.evaluate import development_tasks
from finetune.selective_facts.provenance import write,digest
from finetune.selective_facts_correction.checks import normalize,diagnose
from igt.agents.intent_parser import extract_contract
from eval.contract_quality.harness import prepare,probe

def main():
    path=ROOT/'kaggle_outputs/parsing/selective_facts_scopefix1/selective_facts_scopefix_eval.zip'
    with zipfile.ZipFile(path) as z:
        seal=json.loads(z.read('bundle.json'))
        if z.testzip() is not None or not seal['complete']:raise ValueError('Incomplete bundle')
        for n,sha in seal['sha256'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=sha:raise ValueError('Changed saved record')
        rows={r['task_id']:r for n in z.namelist() if n.startswith('tasks/') and (r:=json.loads(z.read(n)))}
    cases={};gold_flags=[]
    for task,cohort,root in development_tasks():
        before=pd.read_csv(root/task['data_file']);saved=rows[task['id']]
        norm=normalize(saved['raw_output']);issues=diagnose(norm['extracted'],before)
        score={k:saved[k] for k in ['all_correct_accepted','any_wrong_accepted']}
        if norm['changes']:
            if norm['extracted']['status']!='ok':score=dict(all_correct_accepted=False,any_wrong_accepted=False)
            elif cohort=='synthetic':score=probe(norm['extracted']['contract'],prepare(task,root))
            else:raise ValueError('An original-task normalization needs an additional reference replay implementation')
        elif norm['extracted']['contract']!=saved['predicted_contract']:raise ValueError('Unexpected unlogged contract change')
        if cohort=='synthetic':
            flags=diagnose(extract_contract(json.dumps(task['gold_contract'])),before)
            if flags:gold_flags.append(dict(task_id=task['id'],issues=flags))
        cases[task['id']]=dict(original_correct=saved['all_correct_accepted'],original_wrong=saved['any_wrong_accepted'],normalization=norm['changes'],normalized_correct=score['all_correct_accepted'],normalized_wrong=score['any_wrong_accepted'],issues=issues,correction_would_run=bool(issues))
    result=dict(method='Offline normalization replay and correction-trigger diagnosis only; no new model inference or correction score.',source_bundle_sha256=digest(path),checks_sha256=digest(ROOT/'finetune/selective_facts_correction/checks.py'),
        before_correct=sum(c['original_correct'] for c in cases.values()),after_normalization_correct=sum(c['normalized_correct'] for c in cases.values()),after_normalization_wrong=sum(c['normalized_wrong'] for c in cases.values()),
        normalization_tasks=[tid for tid,c in cases.items() if c['normalization']],correction_tasks=[tid for tid,c in cases.items() if c['correction_would_run']],
        passing_tasks_flagged=[tid for tid,c in cases.items() if c['original_correct'] and c['issues']],
        remaining_unflagged_failures=[tid for tid,c in cases.items() if not c['normalized_correct'] and not c['issues']],synthetic_gold_flags=gold_flags,cases=cases)
    write(ROOT/'results/selective_facts_analysis/correction1/offline_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
