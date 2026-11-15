"""Offline screening of contract-check-v1 on saved contracts (CPU only, no model calls).

Runs the input-only diagnostics on every contract saved by the completed combined 3B runs (selective facts,
tools) and the gold run. Reference verdicts are read ONLY here, after diagnosis, to score how often a check
fires on reference-incompatible versus reference-compatible contracts. Gold contracts measure false alarms.

    .venv/bin/python tools/analyze_contract_diagnostics.py [--out results/contract_check_v1/offline_screen.json]
"""
import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import pandas as pd  # noqa: E402
from eval.stage2_selective.runner import load_tasks  # noqa: E402
from eval.stage2_contract_check.diagnostics import diagnose, summary_line  # noqa: E402

BUNDLES = {
    'selective': ROOT / 'kaggle_outputs/stage2/combined_3b_v1/stage2_combined_3b_selective_proposed.zip',
    'tools': ROOT / 'kaggle_outputs/stage2/combined_3b_v1/stage2_combined_3b_tools_proposed.zip',
    'gold': ROOT / 'kaggle_outputs/stage2/combined_gold_v1/stage2_combined_gold_3b.zip',
}


def read_bundle(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for name in names:
            parts = name.split('/')
            if len(parts) >= 2 and parts[-2] == 'tasks' and name.endswith('.json'):
                task = json.loads(z.read(name))
                tid = task['record']['task_id']
                prefix = '/'.join(parts[:-2])
                cname = (prefix + '/' if prefix else '') + f'contracts/{tid}.json'
                contract = json.loads(z.read(cname)) if cname in names else None
                out[tid] = dict(record=task['record'], contract=contract)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'results/contract_check_v1/offline_screen.json')
    args = ap.parse_args()
    tasks = {t['id']: t for _, t in load_tasks('test')}
    data = {f: pd.read_csv(ROOT / f) for f in {t['data_file'] for t in tasks.values()}}
    report = dict(version='contract-check-v1', note='Reference verdicts used only for offline scoring.', methods={})
    for method, path in BUNDLES.items():
        bundle = read_bundle(path)
        rows = []
        for tid in sorted(bundle):
            rec, saved = bundle[tid]['record'], bundle[tid]['contract']
            extracted = saved['extracted'] if saved else dict(status='missing', contract=None)
            issues = diagnose(tasks[tid]['request'], extracted, data[tasks[tid]['data_file']])
            verdict = (rec.get('parser_reference_verdict') or {}).get('status')
            label = 'gold' if method == 'gold' else ('no_contract' if extracted['status'] != 'ok' else
                                                    'compatible' if verdict == 'PASS' else 'incompatible')
            rows.append(dict(task_id=tid, label=label, final_correct=rec['final_correct'],
                             initial_correct=rec['initial_correct'], outcome=rec['outcome'], issues=issues))
        by_label = collections.Counter(r['label'] for r in rows)
        fired = collections.Counter(r['label'] for r in rows if r['issues'])
        proof = collections.Counter(r['label'] for r in rows if any(i['kind'] == 'proof' for i in r['issues']))
        codes = collections.defaultdict(collections.Counter)
        for r in rows:
            for code in {i['code'] for i in r['issues']}:
                codes[code][r['label']] += 1
        report['methods'][method] = dict(
            labels=dict(by_label), any_issue=dict(fired), proof_issue=dict(proof),
            by_code={c: dict(v) for c, v in sorted(codes.items())},
            fires_on_wrong_accept=[r['task_id'] for r in rows if r['issues'] and r['outcome'] == 'ACCEPT'
                                   and not r['final_correct']],
            fires_on_final_correct=[r['task_id'] for r in rows if r['issues'] and r['final_correct']],
            proof_on_final_correct=[r['task_id'] for r in rows if r['final_correct']
                                    and any(i['kind'] == 'proof' for i in r['issues'])],
            missed_incompatible=[r['task_id'] for r in rows if r['label'] == 'incompatible' and not r['issues']],
            rows=rows)
        print(f'\n== {method}: labels {dict(by_label)}')
        print(f'   any issue: {dict(fired)}   proof: {dict(proof)}')
        for c, v in sorted(codes.items()):
            print(f'   {c:30s} {dict(v)}')
        m = report['methods'][method]
        print('   fires on wrong ACCEPT:', m['fires_on_wrong_accept'])
        print('   fires on finally-correct tasks:', m['fires_on_final_correct'])
        print('   PROOF on finally-correct tasks:', m['proof_on_final_correct'])
        print('   incompatible but silent:', m['missed_incompatible'])
        for r in rows:
            if r['issues'] and r['label'] in ('incompatible', 'gold') or (r['issues'] and r['final_correct']):
                print(f'     {r["task_id"]:18s} [{r["label"]}] {summary_line(r["issues"])}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print('\nWrote', args.out.relative_to(ROOT))


if __name__ == '__main__':
    main()
