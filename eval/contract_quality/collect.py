"""Validate two complete pilot ZIPs, then score locally. Does not extract executable payloads."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from .harness import score,SUITE,load_tasks
from .method import EXPERIMENT, DECODING, runtime_hashes
from .context import messages
import pandas as pd

def read_bundle(path):
    with ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP members')
        plan=json.loads(z.read('plan.json'));done=json.loads(z.read('completion.json'))
        rows=[json.loads(l) for l in z.read('predictions.jsonl').decode().splitlines()]
    if plan['smoke'] or done['smoke'] or not done['complete'] or done['records']!=160 or len(rows)!=160:
        raise ValueError(f'{path}: requires a complete, non-smoke 160-record pilot')
    if plan['experiment']!=EXPERIMENT or plan['split']!='dev' or plan['variants']!=['original','grounded']:
        raise ValueError('Wrong experiment')
    if plan['suite_manifest']!=hashlib.sha256((SUITE/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Suite provenance differs')
    if plan.get('runtime')!=runtime_hashes():raise ValueError('Runtime provenance differs; use notebooks and collector from the same pack')
    if plan.get('decoding')!=DECODING or plan.get('dtype')!='float16' or plan.get('quantized') is not False:
        raise ValueError('Inference settings differ')
    spec=json.loads((SUITE/'adapters.json').read_text())[plan['size']]
    if plan['adapter_hashes']!=spec['sha256'] or plan['base']!=spec['model']:raise ValueError('Adapter differs')
    expected={(t['id'],v) for t in load_tasks() for v in ['original','grounded']}
    if {(r['task_id'],r['variant']) for r in rows}!=expected:raise ValueError('Wrong task/variant coverage')
    tasks={t['id']:t for t in load_tasks()}
    frames={}
    for row in rows:
        task=tasks[row['task_id']]
        name=task['data_file']
        if name not in frames:frames[name]=pd.read_csv(SUITE/name)
        msg=messages(task,frames[name],row['variant'])
        expected_hash=hashlib.sha256(json.dumps(msg,sort_keys=True).encode()).hexdigest()
        if row.get('message_sha256')!=expected_hash:raise ValueError('Parser input provenance differs')
    return plan,rows

def render_report(report):
    lines=['# Parser information and semantic checks', '',
           'Exploratory results on 80 synthetic draft development tasks per variant. Independent review is pending.', '',
           'Correct-output withheld counts tasks where at least one known-correct implementation was not accepted. '
           'Wrong acceptance counts tasks where at least one tested wrong output passed. These counts can overlap; '
           'they are parser diagnostics, not Stage 2 final-outcome counts.', '']
    for name,s in report['summary'].items():
        lines += [f'## {name}', '',
            f"Correct implementations accepted on all probes: {s['correct_accepted']}/{s['expected']} tasks.",
            f"Correct output withheld: {s['correct_output_withheld']} tasks, including {s['invalid']} invalid contracts.",
            f"Tasks with a wrong output accepted: {s['tasks_with_wrong_acceptance']}.",
            f"Accepted all correct probes and rejected all wrong probes: {s['joint_success']}.",
            f"Inference: {s['tokens']} tokens; {s['inference_seconds']:.1f} seconds.", '']
    for size in ['1_5b','3b']:
        old=report['summary'][size+'_original'];new=report['summary'][size+'_grounded']
        improved=new['correct_accepted']>old['correct_accepted']
        wrong_ok=new['tasks_with_wrong_acceptance']<=old['tasks_with_wrong_acceptance']
        lines += [f'## {size}: effect of supplying the profile','',
            f"Change in correct acceptance: {new['correct_accepted']-old['correct_accepted']:+d} tasks.",
            f"Change in tasks with wrong acceptance: {new['tasks_with_wrong_acceptance']-old['tasks_with_wrong_acceptance']:+d}.",
            ('The draft counts meet the screening direction; review errors and annotations before selection.'
             if improved and wrong_ok else 'The draft counts do not meet the planned improvement criterion; inspect failures before choosing a training format.'),'']
    lines += ['## Failures to inspect','']
    for name,rows in report['rows'].items():
        failures=[r for r in rows if not r['all_correct_accepted'] or r['any_wrong_accepted']]
        lines += [f'### {name}', '']
        for r in failures:
            detail=r.get('parse_error') or '; '.join(str(v.get('failures',[])) for v in r.get('positive_verdicts',[]) if v['status']!='PASS')
            lines += [f"- {r['task_id']}: correct withheld={not r['all_correct_accepted']}; wrong accepted={r['any_wrong_accepted']}. {detail}"]
        if not failures:lines += ['No failures on the tested probes.']
        lines += ['']
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('bundles',type=Path,nargs=2);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    loaded=[read_bundle(p) for p in a.bundles]
    if {p['size'] for p,r in loaded}!={'1_5b','3b'}:raise ValueError('Need one complete bundle for each adapter size')
    rows=[dict(r,variant=p['size']+'_'+r['variant']) for p,rs in loaded for r in rs]
    if a.out.exists() or a.out.with_suffix('.md').exists():raise ValueError('Output already exists; choose a new path')
    report=score(rows);report['plans']=[p for p,r in loaded]
    if not all(s['complete'] for s in report['summary'].values()):raise ValueError('Incomplete score')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():raise ValueError('Output already exists; choose a new path')
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    a.out.with_suffix('.md').write_text(render_report(report))
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
