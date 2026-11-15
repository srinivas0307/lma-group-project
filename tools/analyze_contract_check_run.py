"""Paired analysis of a contract-check-v1 Kaggle bundle against the completed combined 3B control.

    .venv/bin/python tools/analyze_contract_check_run.py --method selective --checked path/to/stage2_contract_check_selective_checked.zip

Reports correct / incorrect-accepted / withheld for both, per-task gains and losses, check decisions, extra
parser tokens, and whether the checked run's FIRST parse reproduced the control contract (if it did not,
differences are partly runtime nondeterminism, which the report states).
"""
import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.analyze_contract_diagnostics import BUNDLES, read_bundle  # noqa: E402


def score(rows):
    return dict(correct=sum(r['final_correct'] for r in rows),
                incorrect_accepted=sum(r['outcome'] == 'ACCEPT' and not r['final_correct'] for r in rows),
                withheld=sum(r['outcome'] != 'ACCEPT' for r in rows),
                tokens=sum(r['tokens'] for r in rows), parser_tokens=sum(r['parser_tokens'] for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['selective', 'tools'], required=True)
    ap.add_argument('--checked', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    control = read_bundle(BUNDLES[a.method])
    checked = read_bundle(a.checked)
    with zipfile.ZipFile(a.checked) as z:
        plan = json.loads(next(z.read(n) for n in z.namelist() if n.endswith('plan.json')))
    if plan.get('experiment') != 'stage2-contract-check-3b-fp16-v1' or plan.get('arm') != 'checked' \
            or plan.get('method') != a.method or plan.get('smoke') or plan.get('split') != 'test':
        raise SystemExit('Not a complete checked-arm test bundle for this method')
    if set(control) != set(checked):
        raise SystemExit('Task sets differ')
    c_rows = [control[t]['record'] for t in sorted(control)]
    k_rows = [checked[t]['record'] for t in sorted(checked)]
    same_first = []
    for t in sorted(checked):
        first = (checked[t]['contract'] or {}).get('first_extracted') or (checked[t]['contract'] or {}).get('extracted')
        ctrl = (control[t]['contract'] or {}).get('extracted')
        same_first.append(bool(first and ctrl and first.get('contract') == ctrl.get('contract')
                               and first.get('status') == ctrl.get('status')))
    gained = [t for t in sorted(checked) if checked[t]['record']['final_correct'] and not control[t]['record']['final_correct']]
    lost = [t for t in sorted(checked) if control[t]['record']['final_correct'] and not checked[t]['record']['final_correct']]
    decisions = collections.Counter(r.get('check_decision') for r in k_rows)
    report = dict(method=a.method, control=score(c_rows), checked=score(k_rows), gained=gained, lost=lost,
                  first_parse_reproduced=f'{sum(same_first)}/{len(same_first)}', decisions=dict(decisions),
                  corrections_attempted=sum(bool(r.get('check_correction_attempted')) for r in k_rows),
                  check_tokens=sum(r.get('check_tokens', 0) for r in k_rows),
                  per_task=[dict(task_id=r['task_id'], decision=r.get('check_decision'),
                                 first_issues=r.get('check_first_issue_codes'),
                                 remaining=r.get('check_remaining_issue_codes'),
                                 control=control[r['task_id']]['record']['outcome'] + ('/correct' if control[r['task_id']]['record']['final_correct'] else ''),
                                 checked=r['outcome'] + ('/correct' if r['final_correct'] else ''))
                            for r in k_rows if r.get('check_decision') not in (None, 'no_issue')])
    print(json.dumps({k: v for k, v in report.items() if k != 'per_task'}, indent=1))
    for row in report['per_task']:
        print(f"  {row['task_id']:18s} {row['decision']:32s} {row['control']:>16s} -> {row['checked']:<16s} {row['first_issues']}")
    out = a.out or ROOT / f'results/contract_check_v1/{a.method}_paired.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print('Wrote', out)


if __name__ == '__main__':
    main()
