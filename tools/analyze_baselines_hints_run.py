"""Analysis of a stage2-baselines-hints bundle (baseline 2, baseline 3, no-contract-to-generator).

    .venv/bin/python tools/analyze_baselines_hints_run.py --run path/to/stage2_baselines_hints_<run>.zip [--run ...]

For each bundle: correct / incorrect accepted / withheld, parse escalations, first-attempt correctness, tokens, and
paired gains/losses against
  * the matching current-stack system (proposed with repair hints, or the no-contract runs for baseline 3), and
  * the same system in the historical six-system run (clean_v1: 7B NF4 repair, a1-v4, older parsers).
Writes results/baselines_hints_v1/<run>.json.
"""
import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'stage2-baselines-hints-3b-fp16-v1'
HINTS = ROOT / 'kaggle_outputs/stage2/repair_hints_v1'
CLEAN = ROOT / 'kaggle_outputs/stage2/clean_v1'
OURS = ROOT / 'kaggle_outputs/stage2/baselines_hints_v1'
HISTORICAL = {'baseline2_repair': CLEAN / 'stage2_02_baseline2_repair_bundle.zip',
              'baseline3_repair': CLEAN / 'stage2_03_baseline3_repair_bundle.zip',
              'no_contract_to_generator': CLEAN / 'stage2_06_no_contract_to_generator_bundle.zip'}
CURRENT = {'baseline2': {'proposed_selective': HINTS / 'stage2_repair_hints_selective.zip',
                         'proposed_tools': HINTS / 'stage2_repair_hints_tools.zip'},
           'baseline3': {'no_contract_selective': OURS / 'stage2_baselines_hints_no_contract_selective.zip',
                         'no_contract_tools': OURS / 'stage2_baselines_hints_no_contract_tools.zip',
                         'proposed_selective': HINTS / 'stage2_repair_hints_selective.zip'},
           'no_contract_selective': {'proposed_selective': HINTS / 'stage2_repair_hints_selective.zip'},
           'no_contract_tools': {'proposed_tools': HINTS / 'stage2_repair_hints_tools.zip'}}


def load(path):
    plan, rows = None, {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            parts = name.split('/')
            if parts[-1] == 'plan.json' and 'tasks' not in parts:
                plan = json.loads(z.read(name))
            if len(parts) >= 2 and parts[-2] == 'tasks' and name.endswith('.json'):
                r = json.loads(z.read(name))['record']
                if r.get('split', 'test') == 'test':
                    rows[r['task_id']] = r
    return plan, rows


def outcome(r):
    if r['outcome'] == 'ACCEPT':
        return 'correct' if r['final_correct'] else 'incorrect_accepted'
    return 'withheld'


def score(rows):
    c = collections.Counter(outcome(r) for r in rows.values())
    return dict(correct=c['correct'], incorrect_accepted=c['incorrect_accepted'], withheld=c['withheld'],
                parse_escalations=sum(r.get('stage_ended') == 'parse' for r in rows.values()),
                initial_correct=sum(bool(r.get('initial_correct')) for r in rows.values()),
                tokens=sum(r.get('tokens', 0) for r in rows.values()))


def paired(rows, other):
    ids = sorted(set(rows) & set(other))
    return dict(tasks=len(ids),
                only_this_correct=[t for t in ids if rows[t]['final_correct'] and not other[t]['final_correct']],
                only_other_correct=[t for t in ids if other[t]['final_correct'] and not rows[t]['final_correct']])


def analyze(path):
    plan, rows = load(path)
    if not plan or plan.get('experiment') != EXPERIMENT or plan.get('smoke') or plan.get('split') != 'test':
        raise SystemExit(f'{path}: not a complete {EXPERIMENT} test bundle')
    name = plan['common']['baselines_run']
    if len(rows) != 100:
        raise SystemExit(f'{path}: expected 100 test tasks, found {len(rows)}')
    report = dict(run=name, system=plan['preset'], score=score(rows), comparisons={})
    if any('check_decision' in r for r in rows.values()):
        report['contract_check'] = dict(collections.Counter(r.get('check_decision') for r in rows.values()))
        report['reference_compatible_contracts'] = sum(
            (r.get('parser_reference_verdict') or {}).get('status') == 'PASS' for r in rows.values())
    hinted = collections.Counter()
    for r in rows.values():
        hinted[r.get('repair_version', '')] += 1
    report['repair_versions'] = dict(hinted)
    refs = dict(CURRENT.get(name, {}))
    refs['historical_clean_v1'] = HISTORICAL[plan['preset']]
    for label, ref in refs.items():
        if not ref.exists():
            report['comparisons'][label] = 'not available yet: ' + str(ref.relative_to(ROOT))
            continue
        _, other = load(ref)
        report['comparisons'][label] = dict(score=score(other), **paired(rows, other))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, action='append', required=True)
    a = ap.parse_args()
    out_dir = ROOT / 'results/baselines_hints_v1'
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in a.run:
        report = analyze(path)
        s = report['score']
        print(f"== {report['run']} ({report['system']}): {s['correct']} / {s['incorrect_accepted']} / {s['withheld']}"
              f"  parse-escalations {s['parse_escalations']}  first-attempt correct {s['initial_correct']}  tokens {s['tokens']}")
        for label, c in report['comparisons'].items():
            if isinstance(c, str):
                print(f'   vs {label}: {c}')
                continue
            o = c['score']
            print(f"   vs {label}: {o['correct']} / {o['incorrect_accepted']} / {o['withheld']}; "
                  f"only this correct {len(c['only_this_correct'])}, only {label} correct {len(c['only_other_correct'])}")
        (out_dir / f"{report['run']}.json").write_text(json.dumps(report, indent=1))
    print('Wrote', out_dir)


if __name__ == '__main__':
    sys.exit(main())
