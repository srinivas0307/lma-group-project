"""Paired analysis of a repair-hints-v1 Kaggle bundle against its completed baseline.

    .venv/bin/python tools/analyze_repair_hints_run.py --pipeline gold --run path/to/stage2_repair_hints_gold.zip

Baselines: gold -> combined gold 3B (84/0/16); selective/tools -> contract-check checked runs (76/1/23, 74/1/25).
Reports correct / incorrect-accepted / withheld, per-task gains and losses, whether the generator's initial code
reproduced the baseline (only repair changed, so it should), hint/lint usage and tokens. For selective/tools it
also reports an offline admission-gate variant (withhold when contract-check evidence remained unresolved after
the correction turn), computed from the recorded fields; it was designed after seeing these test tasks.
"""
import argparse
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

BASELINES = {
    'gold': ROOT / 'kaggle_outputs/stage2/combined_gold_v1/stage2_combined_gold_3b.zip',
    'selective': ROOT / 'kaggle_outputs/stage2/contract_check_v1/stage2_contract_check_selective_checked.zip',
    'tools': ROOT / 'kaggle_outputs/stage2/contract_check_v1/stage2_contract_check_tools_checked.zip',
}


def load(path):
    tasks, plan = {}, None
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            parts = name.split('/')
            if parts[-1] == 'plan.json' and (len(parts) == 1 or parts[-2] != 'tasks'):
                plan = json.loads(z.read(name))
            if len(parts) >= 2 and parts[-2] == 'tasks' and name.endswith('.json'):
                d = json.loads(z.read(name))
                tasks[d['record']['task_id']] = d
    return plan, tasks


def outcome(r, gate=False):
    if gate and r.get('check_remaining_issue_codes') and r['outcome'] == 'ACCEPT':
        return 'withheld'
    if r['outcome'] == 'ACCEPT':
        return 'correct' if r['final_correct'] else 'incorrect_accepted'
    return 'withheld'


def score(rows, gate=False):
    c = collections.Counter(outcome(r, gate) for r in rows)
    return dict(correct=c['correct'], incorrect_accepted=c['incorrect_accepted'], withheld=c['withheld'],
                tokens=sum(r['tokens'] for r in rows))


def code_hash(trace):
    return hashlib.sha256(trace['initial_code'].encode()).hexdigest() if trace and trace.get('initial_code') else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', choices=list(BASELINES), required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    plan, run = load(a.run)
    _, base = load(BASELINES[a.pipeline])
    if not plan or plan.get('experiment') != 'stage2-repair-hints-3b-fp16-v1' or plan.get('smoke') \
            or plan.get('split') != 'test' or plan.get('method') != a.pipeline:
        raise SystemExit('Not a complete repair-hints test bundle for this pipeline')
    if set(run) != set(base):
        raise SystemExit('Task sets differ')
    ids = sorted(run)
    rr, br = [run[t]['record'] for t in ids], [base[t]['record'] for t in ids]
    same_initial = sum(code_hash(run[t]['trace']) == code_hash(base[t]['trace']) for t in ids
                       if run[t]['trace'] or base[t]['trace'])
    with_code = sum(bool(run[t]['trace'] or base[t]['trace']) for t in ids)
    gained = [t for t in ids if run[t]['record']['final_correct'] and not base[t]['record']['final_correct']]
    lost = [t for t in ids if base[t]['record']['final_correct'] and not run[t]['record']['final_correct']]
    newly_wrong = [t for t in ids if outcome(run[t]['record']) == 'incorrect_accepted'
                   and outcome(base[t]['record']) != 'incorrect_accepted']
    lints = collections.Counter()
    hinted = 0
    for t in ids:
        events = (run[t]['trace'] or {}).get('events', [])
        shown = [e for e in events if e['event'] == 'repair_hints']
        hinted += bool(shown)
        for e in shown:
            lints.update(e.get('lints', []))
    report = dict(pipeline=a.pipeline, baseline=score(br), hints=score(rr), gained=gained, lost=lost,
                  newly_incorrect_accepted=newly_wrong,
                  initial_code_reproduced=f'{same_initial}/{with_code}',
                  tasks_with_hints=hinted, lints_shown=dict(lints),
                  hint_errors=sum(e['event'] == 'repair_hints_error' for t in ids
                                  for e in (run[t]['trace'] or {}).get('events', [])))
    if a.pipeline != 'gold':
        report['offline_gate_unresolved_evidence'] = dict(
            baseline=score(br, gate=True), hints=score(rr, gate=True),
            note='Withhold an accepted result when contract-check issues remained after the correction turn; '
                 'designed after inspecting these test tasks (diagnostic only).')
    report['per_task'] = [dict(task_id=t, baseline=outcome(base[t]['record']), hints=outcome(run[t]['record']),
                               attempts=run[t]['record'].get('attempts'))
                          for t in ids if outcome(base[t]['record']) != outcome(run[t]['record'])]
    print(json.dumps({k: v for k, v in report.items() if k != 'per_task'}, indent=1))
    for row in report['per_task']:
        print(f"  {row['task_id']:18s} {row['baseline']:>18s} -> {row['hints']}")
    out = a.out or ROOT / f'results/repair_hints_v1/{a.pipeline}_paired.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print('Wrote', out)


if __name__ == '__main__':
    main()
