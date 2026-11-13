"""Validate two complete pilot ZIPs, then score locally. Does not extract executable payloads."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from .harness import score,SUITE,load_tasks

def read_bundle(path):
    with ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP members')
        plan=json.loads(z.read('plan.json'));done=json.loads(z.read('completion.json'))
        rows=[json.loads(l) for l in z.read('predictions.jsonl').decode().splitlines()]
    if plan['smoke'] or done['smoke'] or not done['complete'] or done['records']!=160 or len(rows)!=160:
        raise ValueError(f'{path}: requires a complete, non-smoke 160-record pilot')
    if plan['experiment']!='cq-draft-v1' or plan['split']!='dev' or plan['variants']!=['original','grounded']:
        raise ValueError('Wrong experiment')
    if plan['suite_manifest']!=hashlib.sha256((SUITE/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Suite provenance differs')
    spec=json.loads((SUITE/'adapters.json').read_text())[plan['size']]
    if plan['adapter_hashes']!=spec['sha256'] or plan['base']!=spec['model']:raise ValueError('Adapter differs')
    expected={(t['id'],v) for t in load_tasks() for v in ['original','grounded']}
    if {(r['task_id'],r['variant']) for r in rows}!=expected:raise ValueError('Wrong task/variant coverage')
    return plan,rows

def main():
    p=argparse.ArgumentParser();p.add_argument('bundles',type=Path,nargs=2);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    loaded=[read_bundle(p) for p in a.bundles]
    if {p['size'] for p,r in loaded}!={'1_5b','3b'}:raise ValueError('Need one complete bundle for each adapter size')
    rows=[dict(r,variant=p['size']+'_'+r['variant']) for p,rs in loaded for r in rs]
    report=score(rows);report['plans']=[p for p,r in loaded]
    if not all(s['complete'] for s in report['summary'].values()):raise ValueError('Incomplete score')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists():raise ValueError('Output already exists; choose a new path')
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
